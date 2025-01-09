import os
from shutil import copyfile

# You only need to change this line to your dataset download path


import pathlib
working_directory = '/home/zwb/zx/Datasets/Occluded_REID'
def find_filenames(path: pathlib.Path, suffix: str):
    return list(path.rglob('*'+suffix))

#query
query_path = working_directory + '/whole_body_images'
query_save_path = working_directory + '/whole_body_query'

if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

files=find_filenames(pathlib.Path(query_path), '.tif')
for name in files:
    name=str(name)
    if not name[-3:]=='tif':
        continue
    ID  = name.split('/')
    src_path = query_path + '/' +ID[-2]+'/'+ID[-1]
    save_path=query_save_path+'/'+ID[-1]
    copyfile(src_path, save_path)

gallery_path = working_directory + '/occluded_body_images'
gallery_save_path = working_directory + '/occlude_body_gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for name in files:
    name=str(name)
    if not name[-3:]=='tif':
        continue
    ID  = name.split('/')
    src_path = gallery_path + '/' +ID[-2]+'/'+ID[-1]
    save_path=gallery_save_path+'/'+ID[-1]
    copyfile(src_path, save_path)