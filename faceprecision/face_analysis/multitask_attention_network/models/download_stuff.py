import argparse
import gdown

def main():
    parser = argparse.ArgumentParser(description="Download a file using gdown.")
    parser.add_argument('url', type=str, help='The URL of the file to download.')
    parser.add_argument('save_name', type=str, help='The name to save the file as.')

    args = parser.parse_args()

    # Download the file
    gdown.download(args.url, args.save_name, quiet=False)

if __name__ == "__main__":
    main()
