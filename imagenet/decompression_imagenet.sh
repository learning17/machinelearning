#!/bin/bash
# usage:
# ./download_imagenet.sh [dir name]
set -e

if [ -z "$1" ]; then
  echo "Usage: download_imagenet.sh [dir name]"
  exit
fi

OUTDIR="$1"
echo "files dir: $OUTDIR"

SYNSETS_FILE="${OUTDIR}/imagenet_lsvrc_2015_synsets.txt"
BBOX_TAR_BALL="${OUTDIR}/ILSVRC2012_bbox_train_v2.tar.gz"
VALIDATION_TARBALL="${OUTDIR}/ILSVRC2012_img_val.tar"
TRAIN_TARBALL="${OUTDIR}/ILSVRC2012_img_train.tar"
if [ ! -e $SYNSETS_FILE ] || [ ! -e $BBOX_TAR_BALL ] || [ ! -e $VALIDATION_TARBALL ] || [ ! -e $TRAIN_TARBALL ]; then
  echo "file does not exist..."
  exit
fi

#ImageNet bounding boxes
BBOX_DIR="${OUTDIR}/bounding_boxes/"
if [ ! -d $BBOX_DIR ]; then
  echo "Uncompressing ILSVRC2012_bbox_train_v2.tar.gz ..."
  mkdir -p "${BBOX_DIR}"
  tar xzf "${BBOX_TAR_BALL}" -C "${BBOX_DIR}"
else
  echo "bounding_boxes exist..."
fi

#ImageNet 2012 validation dataset
VAL_DIR="${OUTDIR}/validation/"
if [ ! -d $VAL_DIR ]; then
  echo "Uncompressing ILSVRC2012_img_val.tar ..."
  mkdir -p "${VAL_DIR}"
  tar xf "${VALIDATION_TARBALL}" -C "${VAL_DIR}"
else
  echo "validation exist..."
fi

TRAIN_DIR="${OUTDIR}/train/"
if [ ! -d $TRAIN_DIR ]; then
  echo "Uncompressing ILSVRC2012_img_train.tar ..."
  mkdir -p "${TRAIN_DIR}"
  cd "${TRAIN_DIR}/"
  while read SYNSET; do
    echo "Processing: ${SYNSET}"
    mkdir -p "${TRAIN_DIR}/${SYNSET}/"
    rm -rf "${TRAIN_DIR}/${SYNSET}/*"
    tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
    tar xf "${SYNSET}.tar" -C "${TRAIN_DIR}/${SYNSET}/"
    rm -f "${SYNSET}.tar"
    echo "Finished processing: ${SYNSET}"
  done < "${SYNSETS_FILE}"
else
  echo "train exist..."
fi
