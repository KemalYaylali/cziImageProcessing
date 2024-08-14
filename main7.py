import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
import torchvision
from skimage import measure, morphology, feature, filters, exposure, img_as_float32, segmentation
from scipy import ndimage
from sklearn.cluster import DBSCAN
import argparse
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from aicspylibczi import CziFile
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 512)
        self.up1 = self.up(1024, 256)
        self.up2 = self.up(512, 128)
        self.up3 = self.up(256, 64)
        self.up4 = self.up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1[0](x5)
        x = torch.cat([x, self.crop(x4, x)], dim=1)
        x = self.up1[1](x)

        x = self.up2[0](x)
        x = torch.cat([x, self.crop(x3, x)], dim=1)
        x = self.up2[1](x)

        x = self.up3[0](x)
        x = torch.cat([x, self.crop(x2, x)], dim=1)
        x = self.up3[1](x)

        x = self.up4[0](x)
        x = torch.cat([x, self.crop(x1, x)], dim=1)
        x = self.up4[1](x)

        logits = self.outc(x)
        return logits

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels):
        return nn.ModuleList([
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2),
            self.double_conv(in_channels, out_channels)
        ])


class CZIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_czi(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        image_tensor = self.image_transform(image_pil)
        mask_tensor = self.mask_transform(mask_pil)

        return image_tensor, mask_tensor


def log_memory_usage():
    process = psutil.Process(os.getpid())
    logging.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def read_czi(file_path, scale_factor=0.5):
    try:
        logging.debug(f"Attempting to read CZI file: {file_path}")
        czi = CziFile(file_path)
        image_data = czi.read_mosaic(C=0, scale_factor=scale_factor)
        image_array = np.squeeze(image_data)
        image_array = img_as_float32(image_array)
        logging.debug(f"Successfully read CZI file: {file_path}")
        return image_array
    except Exception as e:
        logging.error(f"Error reading image {file_path}: {e}")
        logging.error(traceback.format_exc())
        return None


def preprocess(image, sigma=1, contrast_method='adapt_hist_eq'):
    image = img_as_float32(image)
    denoised = filters.gaussian(image, sigma=sigma)

    if contrast_method == 'adapt_hist_eq':
        enhanced = exposure.equalize_adapthist(denoised)
    elif contrast_method == 'contrast_stretch':
        p2, p98 = np.percentile(denoised, (2, 98))
        enhanced = exposure.rescale_intensity(denoised, in_range=(p2, p98))
    else:
        enhanced = denoised

    return enhanced


def segment_with_unet(image, model):
    image = img_as_float32(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).float().unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    mask = torch.sigmoid(output).squeeze().numpy() > 0.5
    return mask


def watershed_segmentation(image):
    distance = ndimage.distance_transform_edt(image)

    # Try the newer version first, if it fails, use the older version
    try:
        local_max = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max.T)] = True
    except TypeError:
        # Older version of skimage
        local_max_mask = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=image, indices=False)

    markers = measure.label(local_max_mask)
    labels = segmentation.watershed(-distance, markers, mask=image)
    return labels


def advanced_adipose_count(segmented_image):
    labels = watershed_segmentation(segmented_image)
    regions = measure.regionprops(labels)
    logging.debug(f"Total regions found: {len(regions)}")

    valid_regions = [region for region in regions if 100 < region.area < 10000 and region.eccentricity < 0.8]
    logging.debug(f"Valid regions after filtering: {len(valid_regions)}")

    if not valid_regions:
        logging.warning("No valid adipose regions found after filtering")
        return 0

    centroids = np.array([region.centroid for region in valid_regions])

    if len(centroids) < 2:
        logging.warning(f"Not enough centroids for clustering. Returning count of valid regions: {len(valid_regions)}")
        return len(valid_regions)

    n_clusters = dbscan_clustering(centroids)
    logging.debug(f"Number of clusters found: {n_clusters}")
    return n_clusters


def dbscan_clustering(coords, eps=3, min_samples=2):
    if len(coords) < min_samples:
        logging.warning(f"Not enough points for DBSCAN clustering. Returning {len(coords)} as cluster count.")
        return len(coords)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters


def process_file(file_path, adipose_model, nuclei_model):
    try:
        logging.info(f"Processing file: {file_path}")
        image_array = read_czi(file_path)
        if image_array is None:
            logging.warning(f"Failed to read file: {file_path}")
            return None

        adipose_model.eval().float()
        nuclei_model.eval().float()

        if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[0] == 1):
            adipose_channel = image_array[0] if image_array.ndim == 3 else image_array
            nuclei_channel = adipose_channel
        elif image_array.ndim == 3 and image_array.shape[0] >= 2:
            adipose_channel = image_array[0]
            nuclei_channel = image_array[1]
        else:
            raise ValueError(f"Unexpected image dimensions: {image_array.shape}")

        logging.debug(f"Adipose channel shape: {adipose_channel.shape}")
        logging.debug(f"Nuclei channel shape: {nuclei_channel.shape}")

        adipose_preprocessed = preprocess(adipose_channel)
        nuclei_preprocessed = preprocess(nuclei_channel)

        logging.debug(f"Preprocessed adipose shape: {adipose_preprocessed.shape}")
        logging.debug(f"Preprocessed nuclei shape: {nuclei_preprocessed.shape}")

        adipose_segmented = segment_with_unet(adipose_preprocessed, adipose_model)
        nuclei_segmented = segment_with_unet(nuclei_preprocessed, nuclei_model)

        logging.debug(f"Segmented adipose shape: {adipose_segmented.shape}")
        logging.debug(f"Segmented nuclei shape: {nuclei_segmented.shape}")

        adipose_count = advanced_adipose_count(adipose_segmented)
        nuclei_count = advanced_nuclei_count(nuclei_segmented)

        adipose_coverage = np.sum(adipose_segmented) / adipose_segmented.size
        nuclei_coverage = np.sum(nuclei_segmented) / nuclei_segmented.size

        logging.info(f"Adipose count: {adipose_count}, coverage: {adipose_coverage:.2f}")
        logging.info(f"Nuclei count: {nuclei_count}, coverage: {nuclei_coverage:.2f}")

        del image_array, adipose_channel, nuclei_channel, adipose_preprocessed, nuclei_preprocessed
        gc.collect()
        log_memory_usage()

        return {
            'file': os.path.basename(file_path),
            'adipose_count': adipose_count,
            'nuclei_count': nuclei_count,
            'adipose_coverage': adipose_coverage,
            'nuclei_coverage': nuclei_coverage,
            'adipose_segmented': adipose_segmented,
            'nuclei_segmented': nuclei_segmented
        }

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        return None


def create_pdf_report(results, output_folder):
    report_path = os.path.join(output_folder, "CZI_Analysis_Report.pdf")
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    for result in results:
        if result is None:
            continue

        file_name = result['file']
        adipose_count = result['adipose_count']
        nuclei_count = result['nuclei_count']
        adipose_coverage = result['adipose_coverage']
        nuclei_coverage = result['nuclei_coverage']

        c.drawString(30, height - 30, f"Analysis Report for {file_name}")
        c.drawString(30, height - 50, f"Adipose Count: {adipose_count}")
        c.drawString(30, height - 70, f"Nuclei Count: {nuclei_count}")
        c.drawString(30, height - 90, f"Adipose Coverage: {adipose_coverage:.2f}")
        c.drawString(30, height - 110, f"Nuclei Coverage: {nuclei_coverage:.2f}")

        adipose_img_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_adipose.png")
        nuclei_img_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_nuclei.png")

        plt.imsave(adipose_img_path, result['adipose_segmented'], cmap='gray')
        plt.imsave(nuclei_img_path, result['nuclei_segmented'], cmap='nipy_spectral')

        c.drawImage(adipose_img_path, 30, height - 400, width=250, height=250, preserveAspectRatio=True, mask='auto')
        c.drawImage(nuclei_img_path, 300, height - 400, width=250, height=250, preserveAspectRatio=True, mask='auto')

        c.drawString(30, height - 420, "Segmented Adipose Tissue")
        c.drawString(300, height - 420, "Segmented Nuclei")

        c.showPage()

    c.save()
    logging.info(f"PDF report saved at {report_path}")


def main():
    print("Starting main function")
    parser = argparse.ArgumentParser(description="Process CZI files with advanced counting methods.")
    parser.add_argument("--input_dir", default=".", help="Directory containing CZI files for analysis")
    parser.add_argument("--output_dir", default=".", help="Directory for output files")
    parser.add_argument("--adipose_model", help="Path to trained U-Net model for adipose tissue")
    parser.add_argument("--nuclei_model", help="Path to trained U-Net model for nuclei")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    args = parser.parse_args()

    logging.info("Starting the CZI processing script")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.adipose_model or not args.nuclei_model:
        logging.error("--adipose_model and --nuclei_model are required")
        return

    try:
        logging.info("Loading models...")
        adipose_model = UNet(n_channels=1, n_classes=1)
        nuclei_model = UNet(n_channels=1, n_classes=1)
        adipose_model.load_state_dict(
            torch.load(args.adipose_model, map_location=torch.device('cpu'), weights_only=True))
        nuclei_model.load_state_dict(torch.load(args.nuclei_model, map_location=torch.device('cpu'), weights_only=True))
        adipose_model.eval().float()
        nuclei_model.eval().float()
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        logging.error(traceback.format_exc())
        return

    czi_files = glob.glob(os.path.join(args.input_dir, '*.czi'))
    logging.info(f"Found {len(czi_files)} CZI files in the input directory")

    if not czi_files:
        logging.warning("No CZI files found in the input directory.")
        return

    results = []
    if args.parallel:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, f, adipose_model, nuclei_model) for f in czi_files]
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        create_pdf_report([result], args.output_dir)  # Create report for each file
                except Exception as e:
                    logging.error(f"Error in parallel processing: {str(e)}")
    else:
        for f in czi_files:
            result = process_file(f, adipose_model, nuclei_model)
            if result:
                results.append(result)
                create_pdf_report([result], args.output_dir)  # Create report for each file
            gc.collect()
            log_memory_usage()

    if results:
        logging.info("\nSummary of results:")
        for result in results:
            logging.info(f"File: {result['file']}")
            logging.info(f"  Adipose count: {result['adipose_count']}")
            logging.info(f"  Nuclei count: {result['nuclei_count']}")
            logging.info(f"  Adipose coverage: {result['adipose_coverage']:.2f}")
            logging.info(f"  Nuclei coverage: {result['nuclei_coverage']:.2f}")
            logging.info("")
    else:
        logging.warning("No results were successfully processed.")

    logging.info("Script execution completed")


if __name__ == "__main__":
    main()