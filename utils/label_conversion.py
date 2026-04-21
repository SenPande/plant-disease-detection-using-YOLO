from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir='data/raw/Dataset pomodori/',
    save_dir='data/raw/Dataset pomodori'
)

# run " rename 's/\@COLOR_Image//' *.txt " after the conversion to normalize the names