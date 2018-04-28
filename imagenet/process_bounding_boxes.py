#!/usr/bin/python
import glob
import os.path
import sys
import xml.etree.ElementTree as ET
import codecs
import logging

class BoundingBox(object):
    pass

def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    return -1

def GetInt(name, root, index=0):
    return int(float(GetItem(name, root, index)))

def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
            break
        index += 1
    return index

def ProcessXMLAnnotation(xml_file):
    try:
        tree = ET.parse(xml_file)
    except Exception:
        logging.error('Failed to parse: %s' % xml_file)
        return None
    root = tree.getroot()
    num_boxes = FindNumberBoundingBoxes(root)
    boxes = list()
    for index in range(num_boxes):
        box = BoundingBox()
        box.xmin = GetInt('xmin', root, index)
        box.ymin = GetInt('ymin', root, index)
        box.xmax = GetInt('xmax', root, index)
        box.ymax = GetInt('ymax', root, index)
        box.width = GetInt('width', root)
        box.height = GetInt('height', root)
        box.filename = GetItem('filename', root) + '.JPEG'
        box.label = GetItem('name', root)

        xmin = float(box.xmin) / float(box.width)
        xmax = float(box.xmax) / float(box.width)
        ymin = float(box.ymin) / float(box.height)
        ymax = float(box.ymax) / float(box.height)
        min_x = min(xmin, xmax)
        max_x = max(xmin, xmax)
        box.xmin_scaled = min(max(min_x, 0.0), 1.0)
        box.xmax_scaled = min(max(max_x, 0.0), 1.0)

        min_y = min(ymin, ymax)
        max_y = max(ymin, ymax)
        box.ymin_scaled = min(max(min_y, 0.0), 1.0)
        box.ymax_scaled = min(max(max_y, 0.0), 1.0)

        boxes.append(box)
    return boxes

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage: process_bounding_boxes.py <dir> <synsets> <bounding_boxes>")
        exit(1)
    logging.basicConfig(filename='process_bounding_boxes.log', format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    labels = set([l.strip() for l in open(sys.argv[2]).readlines()])
    logging.info("labels size:%d" % len(labels))

    xml_files = glob.glob(sys.argv[1] + '/*/*.xml')
    logging.info("xml_files size:%d" % len(xml_files))

    skipped_files = 0
    skipped_boxes = 0
    saved_boxes = 0
    saved_files = 0
    f = codecs.open(sys.argv[3], 'a+', encoding='utf-8')
    for file_index, one_file in enumerate(xml_files):
        label = os.path.basename(os.path.dirname(one_file))
        if label not in labels:
            logging.error("label %s not in %s" % (label,sys.argv[2]))
            skipped_files += 1
            continue

        bboxes = ProcessXMLAnnotation(one_file)
        if bboxes == None:
            logging.error("No bounding boxes found in :%s" % one_file)
            skipped_files += 1
            continue

        found_box = False
        for bbox in bboxes:
            if bbox.label != label and bbox.label in labels:
                logging.error("bbox.label:%s != %s" % (bbox.label,label))
                skipped_boxes += 1
                continue
            if (bbox.xmin_scaled >= bbox.xmax_scaled or bbox.ymin_scaled >= bbox.ymax_scaled):
                logging.error("Illegal scaled filename:%s" % bbox.filename)
                skipped_boxes += 1
                continue
            image_filename = os.path.splitext(os.path.basename(one_file))[0]
            line = "%s.JPEG,%.4f,%.4f,%.4f,%.4f\n" % (image_filename,bbox.xmin_scaled,bbox.ymin_scaled,bbox.xmax_scaled, bbox.ymax_scaled)
            f.write(line)
            saved_boxes += 1
            found_box = True
        if found_box:
            saved_files += 1
        else:
            skipped_files += 1
        if file_index % 5000 == 0:
            logging.info('--> processed %d of %d XML files.' % (file_index + 1, len(xml_files)))
            logging.info('--> skipped %d boxes and %d XML files.' % (skipped_boxes, skipped_files))
    logging.info('Finished processing %d XML files.' % len(xml_files))
    logging.info('Skipped %d XML files not in ImageNet Challenge.' % skipped_files)
    logging.info('Skipped %d bounding boxes not in ImageNet Challenge.' % skipped_boxes)
    logging.info('Wrote %d bounding boxes from %d annotated images.' % (saved_boxes, saved_files))

