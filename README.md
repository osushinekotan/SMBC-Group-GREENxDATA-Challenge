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

## Solution

### Feature Engineering
- raw features
    - 元々数値特徴量
    - `created_at` をパースした特徴量 (yyyy-mm など)
    - `guards` などの categorical 特徴を適当に rank 化 (`{"Helpful": 0, "Unsure": 1, "Harmful": 2}` など)
    - `problems` の個数や、種類別の binary feature
- ordinal encoding
    - 元々の特徴量 
    - `nta` を数値部分と文字列部分にパースしたものなど
- aggregation
    - `boro_ct` や `guards`、`boro_ct`+`created_at__ym` などで以下を mean / std で集約
        - `tree_dbh`
        - rank features (`gurds` や `steward` など)
        - problem binary feautres

### TTA (Test Time Augmentation)
- `created_at` をキーに前後 3,4 日分テストデータを拡張し、最終的に平均をとる (train データの augmentation もしてみたが、特にワークしなかった)

### Model
- `lightgbm` : `num_leaves` と `random_state` を変化させながらモデルを作り、それぞれの予測値の平均を最終的な予測とした
- `xgboost`, `catboost` なども少し試したが、うまくいかなかったため `lightgbm` のみを使用 
- stacking では `num_leaves` を大きめに設定した `lightgbm` を使用。特徴量は別モデルの予測値 (proba + label) と集約特徴量以外の特徴量を使用

### Ensemble
- 各モデルの probability を加重平均する。この時の weight は最適化したものを使用
- probability を下記の後処理でラベル化

### Post Processing
- argmax でラベル化するのではなく、最初に `health=2` 予測確率の閾値を最適化しながらラベル化、次に `health=0` 最後に `health=1` を同様にラベル化するようにした (`src.experiment.optimization.find_optimal_threshold_for_label`)

### まとめ
- TTA 有無や、特徴量や lightgbm のパラメタを色々変えてみつつ数十種類のモデルを作成
- 同様に stacking を実行
- 最終的にそれらの予測値をアンサンブルし、後処理を実行

### うまくいかなかったこと
- validation
    - `stratified k fold` で cross validation をしたが、特徴量作成を fold ごとに実行しなかったので適切な検証スコアを取得できなかったと思う
    - 集約特徴に関して、test には存在するが train に存在しない category があるため、それを validation data で表現できていないので良い cv スコアを得ることができなかった (train test を concat して特徴作成すれば問題ないがルール上 NG 😭)
- transformer
    - カラムを結合し、一つのテキストをみなして `deberta-v3` を fine-tuning したが特にワークしなかった。
    - 今回は `created_at` の値自体が強力なので、その辺りをうまく扱えるようにすればうまくいったかもしれない
- 重複外削除
    - train / test においてそれぞれのデータにしかないカテゴリを欠損などに置き換える
    - どちらかといえばうまくいったが、ルールに抵触する可能性あり
- rolling features
    - `created_at` をキー含めた移動平均集約特徴量
    - aggregation 同様、test にのみ存在する category や `created_at` をうまく扱えなかった。test のみで計算したり、test のみ train + test のデータを使って計算したりしたが not work だった。 
