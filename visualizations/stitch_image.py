import os
from PIL import Image
import math
import argparse
import os
import math
from PIL import Image
from pdf2image import convert_from_path

def stitch_pdfs(image_dir, output_path="stitched_output.pdf", cols=4, padding=10, resize_width=None, dpi=150):
    # Get all .pdf files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".pdf")])
    if not image_files:
        print("No PDF files found in the directory.")
        return

    images = []
    for f in image_files:
        pdf_path = os.path.join(image_dir, f)
        # Convert the first page of the PDF to an image at specified DPI
        rendered = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)[0]
        if resize_width:
            w_percent = resize_width / float(rendered.size[0])
            h_size = int(float(rendered.size[1]) * w_percent)
            rendered = rendered.resize((resize_width, h_size), Image.ANTIALIAS)
        images.append(rendered.convert("RGB"))

    # Compute grid size
    n = len(images)
    rows = math.ceil(n / cols)
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    stitched = Image.new("RGB", (
        cols * max_width + padding * (cols - 1),
        rows * max_height + padding * (rows - 1)
    ), color=(255, 255, 255))  # white background

    for idx, img in enumerate(images):
        x = (idx % cols) * (max_width + padding)
        y = (idx // cols) * (max_height + padding)
        stitched.paste(img, (x, y))

    # Save as a single-page PDF
    stitched.save(output_path, format="PDF", resolution=dpi)
    print(f"Stitched PDF saved to: {output_path}")


def stitch_images(image_dir, output_path="stitched.png", cols=4, padding=10, resize_width=None):
    # Get all .png files, sorted by filename
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".pdf")])
    if not image_files:
        print("No PNG images found in the directory.")
        return

    images = []
    for f in image_files:
        img = Image.open(os.path.join(image_dir, f))
        if resize_width:
            w_percent = resize_width / float(img.size[0])
            h_size = int(float(img.size[1]) * w_percent)
            img = img.resize((resize_width, h_size), Image.ANTIALIAS)
        images.append(img)

    # Grid dimensions
    n = len(images)
    rows = math.ceil(n / cols)
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    stitched = Image.new("RGB", (
        cols * max_width + padding * (cols - 1),
        rows * max_height + padding * (rows - 1)
    ), color=(255, 255, 255))  # white background

    # Paste images into grid
    for idx, img in enumerate(images):
        x = (idx % cols) * (max_width + padding)
        y = (idx // cols) * (max_height + padding)
        stitched.paste(img, (x, y))

    stitched.save(output_path)
    print(f"Stitched image saved to: {output_path}")

if __name__ == "__main__":
    # python stitch_image.py --image_dir visualizations/histograms/[name_of_dir] --output visualizations/histograms/[name_of_dir]/stitched.png --cols 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing .png files")
    parser.add_argument("--output", type=str, default="stitched.png", help="Output image filename")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid")
    parser.add_argument("--resize_width", type=int, default=None, help="Optional width to resize each image")
    args = parser.parse_args()

    # stitch_images(
    #     image_dir=args.image_dir,
    #     output_path=args.output,
    #     cols=args.cols,
    #     resize_width=args.resize_width
    # )

    stitch_pdfs(
        image_dir=args.image_dir,
        output_path=args.output,
        cols=args.cols,
        resize_width=args.resize_width
    )
