from PIL import Image
import sys

def crop_image(image_path, left, upper, right, lower, output_path):
    image = Image.open(image_path)
    width, height = image.size
    cropped_image = image.crop((left, upper, width-right, height-lower))
    cropped_image.save(output_path)
    print(f"Cropped image saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_image.py <input_image_path> <output_image_path>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    # Coordinates for cropping: left, upper, right, lower
    left = 500
    upper = 300
    right = 700
    lower = 290

    crop_image(input_image_path, left, upper, right, lower, output_image_path)

