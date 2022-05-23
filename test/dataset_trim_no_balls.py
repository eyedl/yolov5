import os
import random
import argparse
# data_loc = r'C:\Users\koen_\OneDrive\Documents\.Koen de Raad\- Whitebox Data Science\Projecten\TeamTV\Trained Models\Hockey\detectors\dataset_hockey_ball_detection_txt'

if __name__ == '__main__':
    # def_data = r'C:\Users\koen_\OneDrive\Documents\.Koen de Raad\- Whitebox Data Science\Projecten\TeamTV\Trained Models\Hockey\detectors\dataset_hockey_ball_detection_txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='/home/teamtv/eyedle-training/darknet/data/hockeyball_32tr_6te/obj_train_data', help='path to dataset folder')
    parser.add_argument('-f', '--folder', type=str, default='task', help='folder name to filter between train and tests')
    parser.add_argument('-o', '--output_file', type=str, default='/home/teamtv/eyedle-training/darknet/data/hockeyball_32tr_6te/yolov5_hockeyball_32tr_6te_train_filled_unbalanced.txt', help='txt file to output the data to')
    parser.add_argument('-p', '--pct_empty', type=float, default=0.05, help='percentage of empty files that is needed')
    args = parser.parse_args()

    data_loc = args.data
    empty_label_pct = args.pct_empty
    output_file = args.output_file
    folder_filter = args.folder


    # below we walk through all txt files and check whether they have any label or not
    empty_labels = []
    filled_labels = []
    for root, dirs, files in os.walk(data_loc):
        if root.split(os.sep)[-1].startswith(folder_filter):  # specify the name of the training folders
            if files:
                for file in files:
                    if file.endswith('.txt'):
                        file_loc = os.path.join(root,file)
                        file_size = os.stat(file_loc).st_size
                        if file_size > 0:
                            filled_labels.append(file_loc.replace('.txt', '.png'))
                        else:
                            empty_labels.append(file_loc.replace('.txt', '.png'))

    # making sure everything is randomly shuffled
    random.shuffle(filled_labels)
    random.shuffle(empty_labels)

    # calculating exact number of empty labels needed given a percentage
    len_filled_labels = len(filled_labels)
    len_empty_labels = round((len_filled_labels * empty_label_pct) / (1 - empty_label_pct))

    # adding the right amount of empty labels to the dataset
    combined_labels = filled_labels + empty_labels[:len_empty_labels+1]
    # making sure everything is randomly shuffled
    random.shuffle(combined_labels)

    with open(output_file, 'w+') as f:
        # adding all the filled labels and empty labels together
        for ix, i in enumerate(combined_labels):
            f.write(i+'\n')