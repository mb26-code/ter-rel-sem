#!/bin/bash

#ce script bash permet de copier-coller tout le contenu d'un dossier vers un autre dossier

#vérifie qu'on a bien deux arguments: le dossier source et le dossier destination
if [ "$#" -ne 2 ]; then
  echo "Utilisation : $0 [dossier_source] [dossier_destination]"
  exit 1
fi

SOURCE="$1"
DEST="$2"

#crée le répertoire destination s’il n’existe pas
mkdir -p "$DEST"

#copie tous les fichiers (non récursif)
cp "$SOURCE"/* "$DEST"/

echo "Fichiers copiés de $SOURCE vers $DEST"