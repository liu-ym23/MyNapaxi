#!/usr/bin/env bash
# Rebuild the companion screen-recorder APK from source (javac + d8 + aapt2,
# no gradle needed). Output overwrites ../../bench-recorder.apk.
#
# Requires: Android SDK build-tools (36.x) and platforms/android-36.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BT="${ANDROID_HOME:-$HOME/.local/share/android/sdk}/build-tools/36.0.0"
PLAT="${ANDROID_HOME:-$HOME/.local/share/android/sdk}/platforms/android-36/android.jar"

rm -rf "$HERE/build" && mkdir -p "$HERE/build/classes"
javac --release 11 -cp "$PLAT" -d "$HERE/build/classes" \
    "$HERE"/java/com/napaxi/bench/recorder/*.java
"$BT/d8" --release --lib "$PLAT" --output "$HERE/build" \
    $(find "$HERE/build/classes" -name '*.class')
"$BT/aapt2" link -o "$HERE/build/base.apk" --manifest "$HERE/AndroidManifest.xml" \
    -I "$PLAT" --target-sdk-version 29 --min-sdk-version 26 \
    --version-code 5 --version-name 1.4
python3 - "$HERE" <<'PY'
import sys, zipfile, shutil
here = sys.argv[1]
shutil.copy(f"{here}/build/base.apk", f"{here}/build/withdex.apk")
with zipfile.ZipFile(f"{here}/build/withdex.apk", "a") as z:
    z.write(f"{here}/build/classes.dex", "classes.dex")
PY
"$BT/zipalign" -f 4 "$HERE/build/withdex.apk" "$HERE/build/aligned.apk"
if [ ! -f "$HERE/build/debug.keystore" ]; then
    keytool -genkeypair -keystore "$HERE/build/debug.keystore" -alias bench \
        -storepass android -keypass android -dname "CN=bench" -keyalg RSA -validity 3650
fi
"$BT/apksigner" sign --ks "$HERE/build/debug.keystore" --ks-pass pass:android \
    --out "$HERE/../../bench-recorder.apk" "$HERE/build/aligned.apk"
echo "rebuilt: $HERE/../../bench-recorder.apk"
