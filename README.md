# smbc-group-greenxdata-challenge

## Env
```
cp .env.example .env
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

## Run
final submission : `128-ensemble_v2`

1. 以下の ensemble 対象の experiment を実行する
```
ensemble_exps:
  - experiment=101-tabular_v6
  - experiment=102-tabular_v6
  - experiment=103-tabular_v6
  - experiment=104-tabular_v6
  - experiment=105-tabular_v6
  - experiment=106-tabular_v6
  - experiment=107-tabular_v6
  - experiment=108-tabular_v7
  - experiment=115-tabular_v7
  # - experiment=116-tabular_v7
  # - experiment=117-tabular_v7
  # - experiment=118-tabular_v7
  # - experiment=119-tabular_v7
  - experiment=112-stacking_v4
  - experiment=113-stacking_v4
  - experiment=120-stacking_v4
  - experiment=121-stacking_v4
  - experiment=124-stacking_v4  
```
2. `128-ensemble_v2` を実行する

```
rye run python experiment={EXP_NAME}
```