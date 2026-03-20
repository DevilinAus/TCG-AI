#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="$ROOT_DIR/frontend/assets/cards/my-first-battle"
BASE_URL="https://archives.bulbagarden.net/wiki/Special:Redirect/file"

mkdir -p "$ASSET_DIR"

download() {
  local source_name="$1"
  local output_name="$2"
  local output_path="$ASSET_DIR/$output_name"

  echo "Downloading $source_name -> $output_name"
  curl -fsSL "$BASE_URL/$source_name" -o "$output_path"
}

download "BulbasaurMyFirstBattle.jpg" "bulbasaur.jpg"
download "IvysaurMyFirstBattle.jpg" "ivysaur.jpg"
download "OddishMyFirstBattle.jpg" "oddish.jpg"
download "GloomMyFirstBattle.jpg" "gloom.jpg"
download "ExeggcuteMyFirstBattle.jpg" "exeggcute.jpg"
download "ExeggutorMyFirstBattle.jpg" "exeggutor.jpg"
download "ScytherMyFirstBattle.jpg" "scyther.jpg"

download "CharmanderMyFirstBattle.jpg" "charmander.jpg"
download "CharmeleonMyFirstBattle.jpg" "charmeleon.jpg"
download "VulpixMyFirstBattle.jpg" "vulpix.jpg"
download "NinetalesMyFirstBattle.jpg" "ninetales.jpg"
download "GrowlitheMyFirstBattle.jpg" "growlithe.jpg"
download "ArcanineMyFirstBattle.jpg" "arcanine.jpg"
download "MagmarMyFirstBattle.jpg" "magmar.jpg"

download "PikachuMyFirstBattle.jpg" "pikachu.jpg"
download "RaichuMyFirstBattle.jpg" "raichu.jpg"
download "MagnemiteMyFirstBattle.jpg" "magnemite.jpg"
download "MagnetonMyFirstBattle.jpg" "magneton.jpg"
download "VoltorbMyFirstBattle.jpg" "voltorb.jpg"
download "ElectrodeMyFirstBattle.jpg" "electrode.jpg"
download "ElectabuzzMyFirstBattle.jpg" "electabuzz.jpg"

download "SquirtleMyFirstBattle.jpg" "squirtle.jpg"
download "WartortleMyFirstBattle.jpg" "wartortle.jpg"
download "PoliwagMyFirstBattle.jpg" "poliwag.jpg"
download "PoliwhirlMyFirstBattle.jpg" "poliwhirl.jpg"
download "MagikarpMyFirstBattle.jpg" "magikarp.jpg"
download "GyaradosMyFirstBattle.jpg" "gyarados.jpg"
download "LaprasMyFirstBattle.jpg" "lapras.jpg"

download "PotionScarletViolet188.jpg" "potion.jpg"
download "SwitchScarletViolet194.jpg" "switch.jpg"

download "BasicGrassEnergySVEEnergy1.jpg" "grass_energy.jpg"
download "BasicFireEnergySVEEnergy2.jpg" "fire_energy.jpg"
download "BasicWaterEnergySVEEnergy3.jpg" "water_energy.jpg"
download "BasicLightningEnergySVEEnergy4.jpg" "lightning_energy.jpg"

echo "Download complete."
