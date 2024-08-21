from Segment_Library.Aug20_pipeline import *

# Define a main function which can take two arguments when called from the command line

def main(input_mask_path, output_measurement_path, calibration_factor:float, mask_index: int):
    masks = np.load(input_mask_path, allow_pickle=True)
    length_and_width(masks, mask_index, calibration_factor, output_measurement_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))