# To process CORRUPT EXIF data warning, run this code.
import glob
import piexif


nfiles=0
for filename in glob.iglob('./data/ILSVRC2015/Data/CLS-LOC/train/**/*.JPEG', recursive=True):
    nfiles += 1
    print("About to process file %d, which is %s." %(nfiles, filename))
    piexif.remove(filename)

