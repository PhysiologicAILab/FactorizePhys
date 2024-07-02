import numpy as np
from datetime import datetime, timezone, time
from pathlib import Path

import fnv
import fnv.reduce
import fnv.file

def strip_time(time_string):
    time_string_split = time_string.split(":")
    return ":".join(time_string_split[1:])


class IRDataLoader(object):
    def __init__(self, path, datatype="seq", stop_frame=0) -> None: #type="seq" or "raw_img"

        self.stop_frame = stop_frame

        if "seq" in datatype.lower():
            self.datatype = "seq"
            self.im = fnv.file.ImagerFile(str(path))    # open the file
            self.im.unit = fnv.Unit.TEMPERATURE_FACTORY      # set the desired unit

            if self.stop_frame == 0:
                self.stop_frame = self.im.num_frames
        
        elif "raw" in datatype.lower():
            self.datatype = "raw"
            self.fp_list = sorted(list(path.glob("*.raw")))
            if self.stop_frame == 0:
                self.stop_frame = len(self.fp_list)
            self.im_width = 640
            self.im_height = 512
        else:
            print("*"*50)
            print("!!!!!! Invalid datatype specified")
            print("*"*50)

        self.indx = 0

    def load_next(self):
        if self.indx < self.stop_frame:
            if self.datatype == "seq":
                self.im.get_frame(self.indx)                         # get the next frame
                for f in self.im.frame_info:
                    if f['name'] == "Time":
                        timestamp = f['value']
                        timestamp_time = datetime.strptime(strip_time(timestamp), "%H:%M:%S.%f")
                        timestamp_time = timestamp_time.replace(year=1970)
                im_array = np.array(self.im.final, copy=True).reshape((self.im.height, self.im.width))
                timestamp = timestamp_time.replace(tzinfo=timezone.utc).timestamp()

            else:
                fpath = str(self.fp_list[self.indx])
                im_array = np.fromfile(fpath, dtype=np.uint16, count=self.im_width *
                                            self.im_height).reshape(self.im_height, self.im_width)
                im_array = im_array.astype(np.float32)
                im_array = (im_array * 0.04) - 273.15

            self.indx += 1
        else:
            im_array = None
            # timestamp = None
        # return im_array, timestamp
        return im_array
