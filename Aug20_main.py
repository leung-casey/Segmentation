from Segment_Library.Aug20_pipeline import *

# Define a main function which can take two arguments when called from the command line

def main(input_path, output_figure_path, frame_index: int):
    frames = path_to_frame_array(input_path)
    image = frames_to_image(frames, frame_index)
    masks = create_mask(image)
    show_all_masks(masks, image, output_figure_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))