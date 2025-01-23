import os
from PIL import Image


# Useful for images that are not RGB, such as the ones in the dataset provided by the client that are in palette mode.
def decompress_images(input_directory: str):
    output_directory = f"{input_directory}-decompressed"
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        # Try to open the image file
        try:
            with Image.open(file_path) as img:
                # Check image mode and convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save the decompressed image to the output directory
                output_path = os.path.join(output_directory, filename)
                img.save(output_path)

                print(f"Converted to RGB and saved {filename} to {output_directory}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def main():
    image_directory = "fire_risk_classifier/data/images/ortos2018-RGB-50m"
    decompress_images(image_directory)
