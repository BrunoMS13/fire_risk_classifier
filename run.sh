
cd /tmp

git clone https://github.com/BrunoMS13/fire_risk_classifier.git

mkdir -p fire_risk_classifier/fire_risk_classifier/data/images

cp -r ~/fire_risk_classifier/fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed \
      /tmp/fire_risk_classifier/fire_risk_classifier/data/images

echo "Run stuff..."