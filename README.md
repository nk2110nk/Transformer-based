# Transformer_Negotiator

Transformerを用いた自動交渉AIエージェントの実装です。  
`negmas`上の交渉ドメインを使い、強化学習(PPO)で3者間交渉用のエージェントを学習・評価します。

## リポジトリ構成

```text
.
├── train.py                 # 学習実行スクリプト
├── test_negotiator.py       # 学習済みモデルの評価スクリプト
├── ppo_scratch.py           # PPOの学習ループ
├── policy.py                # Transformer方策・価値関数・RolloutBuffer
├── NegTransformer.py        # Transformer本体
├── envs/                    # 交渉環境、観測、RLNegotiator
├── sao/                     # SAOメカニズム用Negotiator
├── domain/                  # GENIUS形式の交渉ドメイン
├── embeddings/              # 事前計算済み埋め込み
├── data_SMIHT/              # 実験データ
├── summary_tables/          # 集計済み結果
├── data_calculator/         # 結果集計スクリプト
├── results/                 # 学習済みモデル・評価結果の出力先
├── requirements.txt
└── Dockerfile
```

## セットアップ

Python 3.10での実行を想定しています。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<24" "setuptools<66" "wheel<0.41"
pip install -r requirements.txt
```

GPUを使う場合は、環境に合うCUDA版PyTorchが入っていることを確認してください。CPUで動かす場合は、学習・テスト時に`-d cpu`を指定できます。

## Dockerで実行する場合

```bash
docker build -t transformer-negotiator .
docker run --rm -it -v "$PWD:/app" transformer-negotiator bash
```

コンテナ内で以降の学習・テストコマンドを実行します。

## 利用できるドメインと相手エージェント

主に以下のドメイン・エージェント名を指定できます。

```text
ドメイン:
Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A

相手エージェント:
Boulware Linear Conceder TitForTat1 TitForTat2 AgentK HardHeaded Atlas3 AgentGG
```

## 学習方法

基本コマンドは次の形式です。

```bash
python3 train.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```

短めに動作確認したい場合は、`-ts`や`-hr`を小さくします。

```bash
python3 train.py \
  -a Boulware Conceder \
  -i Laptop \
  -d cpu \
  -ts 4096 \
  -hr 256 \
  -bs 64
```

学習結果はデフォルトで次のようなディレクトリに保存されます。

```text
results/<ドメイン名を-で連結>_<エージェント名を-で連結>/<YYYYMMDD-HHMMSS>-TA/
```

その中に`checkpoint.pt`とTensorBoardログが出力されます。

### 主な学習オプション

```text
-a,  --agents             学習で使う相手エージェント名
-i,  --issue              学習で使うドメイン名
-m,  --model              継続学習する既存モデルのディレクトリ
-d,  --device             auto / cuda / cpu
-lr, --learning_rate      学習率。デフォルトは3e-4
-ts, --total_timesteps    総ステップ数。デフォルトは300000
-bs, --batch_size         バッチサイズ。デフォルトは64
-hr, --horizon            1ロールアウトのステップ数。デフォルトは2048
-ec, --entropy_coef       エントロピー係数。デフォルトは0.0
-cr, --clip_range         PPOのclip range。デフォルトは0.2
-s,  --scale              埋め込みサイズ設定。通常はsmall
-r,  --random_train       学習対象のドメイン・相手をランダム化
-do, --decoder_only       decoder-onlyモデルを使う
-dn, --decoder_num        decoderブロック数。デフォルトは1
-sp, --save_path          保存先を直接指定
```

### 継続学習

既存モデルのディレクトリを`-m`に指定します。指定先には`checkpoint.pt`が必要です。

```bash
python3 train.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress \
  -m ./results/Laptop-ItexvsCypress_Boulware-Conceder-Linear-Atlas3/20260321-184856-TA \
  -ts 100000
```

保存先を固定したい場合は`-sp`を使います。

```bash
python3 train.py \
  -a Boulware Conceder \
  -i Laptop \
  -sp ./results/my_experiment
```

## テスト方法

`test_negotiator.py`は、`-m`で指定したディレクトリ直下の`checkpoint.pt`を読み込みます。

```bash
python3 test_negotiator.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A \
  -m ./results/Laptop-ItexvsCypress-IS_BT_Acquisition-Grocery-thompson-Car-EnergySmall_A_Boulware-Conceder-Linear-Atlas3/20260321-184856-TA
```

短く動作確認する例です。

```bash
python3 test_negotiator.py \
  -a Boulware Conceder \
  -i Laptop \
  -m ./results/my_experiment
```

評価結果はモデルディレクトリ配下に保存されます。

```text
<model_dir>/csv/<agent0>-<agent1>/<issue>/det=False_noise=False/*.tsv
```

プロットも保存したい場合は`-p`を付けます。この場合は1回分の交渉を実行し、`img/`配下に画像を保存します。

```bash
python3 test_negotiator.py \
  -a Boulware Conceder \
  -i Laptop \
  -m ./results/my_experiment \
  -p
```

### 主なテストオプション

```text
-a,  --agents        評価で使う相手エージェント名
-i,  --issues        評価で使うドメイン名
-m,  --model         checkpoint.ptを含むモデルディレクトリ
-s,  --scale         埋め込みサイズ設定。通常はsmall
-do, --decoder_only  学習時にdecoder-onlyを使った場合に指定
-dn, --decoder_num   学習時と同じdecoderブロック数
-p,  --plot          交渉ログのプロットを保存
-if, --is_first      先攻設定用フラグ
```

## TensorBoardで学習ログを見る

```bash
tensorboard --logdir ./results
```

ブラウザで表示されたURLを開くと、`train/loss`や`rollout/ep_rew_mean`などを確認できます。

## 埋め込みを作り直す場合

`embeddings/`には事前計算済みの埋め込みが含まれています。新しいドメインを追加した場合など、埋め込みを再生成するには`embedding_model.py`を使います。

事前に`embedding_model.py`内のOpenAI APIキー設定、または環境変数利用への変更を行ってください。

```bash
python3 embedding_model.py
```

出力先は次の通りです。

```text
embeddings/openai/small/<domain>.json
embeddings/openai/i_embs/small/<domain>.json
```

## 結果集計

評価結果を集計するスクリプトは`data_calculator/`にあります。

```bash
python3 data_calculator/summary_data.py
python3 data_calculator/summary_unexpect_data.py
```

集計結果は`summary_tables/`に保存されます。

## 注意点

- `train.py`では`--issue`、`test_negotiator.py`では`--issues`という引数名ですが、どちらも短縮形`-i`で指定できます。
- テスト時の`-m`には、`checkpoint.pt`そのものではなく、`checkpoint.pt`を含むディレクトリを指定してください。
- `decoder_only`や`decoder_num`を変えて学習したモデルを評価する場合、テスト時にも同じ設定を指定してください。
- 依存パッケージを入れる前は、`train.py --help`や`test_negotiator.py --help`もimportエラーになります。先に`pip install -r requirements.txt`を実行してください。
