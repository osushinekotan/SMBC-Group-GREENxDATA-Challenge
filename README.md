# smbc-group-greenxdata-challenge

## Env
```
cp .env.example .env
echo 'export $(cat .env | grep -v ^#)' >> ~/.bashrc
```
```
mkdir resources/inputs
```

## Rye
```
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
```
```
rye sync
```

## Data
`resources/inputs` に `train.csv`, `test.csv`, `submission.csv` を配置する