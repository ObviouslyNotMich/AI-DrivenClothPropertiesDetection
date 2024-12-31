import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

# directory management
import os
import shutil
from datetime import datetime


class Data:
    def __int__(self):
        # initialiseer de locatie van de beelddata

        self.data = "/home/fashion/fashion ai repo/data"
        self.temp_data = "/home/fashion/fashion ai repo/data_temp"
        self.processed_data = "/home/fashion/fashion ai repo/data_processed"

        try:
            os.mkdir(self.temp_data)  # maak folder
        except OSError as error:
            print(error)  # error output

    def sq_cut_img(self):
        tot_start_time = datetime.now()

        # TODO: implement list and increment list method every time there is a corrupted images.
        # TODO: implement list with increment method everytime image successfully splitted.

        for beeld in self.data:

            start_time = datetime.now()  # timer start

            img_path = os.path.join(os.path.join(self.data), beeld)  # huidige beeldpad maken

            # skip corrupted images
            try:
                img = cv2.imread(img_path)  # laad het beeld met cv2.imread

                img = cv2.resize(img, dsize=(5000, 5000), interpolation=cv2.INTER_CUBIC)

                full_path = os.path.join(self.temp_data, beeld[:-4])

                try:
                    os.mkdir(full_path)  # maak folder
                except OSError as error:
                    print(error)  # error output

                hoogte = img.shape[0]
                breedte = img.shape[1]

                hoog = hoogte // 25
                breed = breedte // 25

                for x in range(0, hoogte, hoog):  #
                    for y in range(0, breedte, breed):
                        beeld_gesplitst = img[x:x + hoog, y:y + breed]

                        cv2.imwrite(full_path + '/' + beeld[:13] + '_' + str(y) + '_' + str(x) + '.png',
                                    beeld_gesplitst)

                # time stop
                end_time = datetime.now()
                print(full_path + ' is gemaakt in  ' + str(end_time - start_time) + ' seconden')

                # TIPS: Originele file verwijderen
                self.verwijder_file(img_path)

            except:
                print('Corrupted image: ', img_path)
                pass

        # Totaal process in seconden
        tot_end_time = datetime.now()
        print('Totaal in ' + str(tot_end_time - tot_start_time) + ' seconden')

    def verwijder_file(self, file):
        if os.path.exists(file):
            os.remove(file)  # een file verwijderen
            print(file + ' is verwijderd')
        else:
            print('De file bestaat niet!')

