# レポート課題１
析結果，考察，参考文献リスト，講義の感想をレポートにまとめよ．レポート文書のPDF，作成したプログラム，データファイル（公開データの場合URLのみ記載）等を，ITC-LMSで提出せよ．他者と協力した場合はその番号・氏名を明記．

なお，ChatGPT等AIを活用した場合は，レポートの最初にアプリ名とバージョンを明記し，対話記録を付録として添付し，本文後半に，プロンプトにどういう工夫をしたか，どう役立ち，どこには役立たなかったか，どこに注意が必要だったか，使用してみた感想や評価を書いた上で，自分のためになるような本課題のAI活用を前提とした改良案を提案せよ．なお，虚偽申告や無断転載等が確認されたものは履修成績を不合格とすることがあり，また，内容が不自然・不整合な場合は減点することがあるので注意せよ．


## 【課題】

Hopfield型のニューラルネットをシミュレートするプログラムを作り，まず1種類の5x5の2値(+1/-1)画像を覚えさせ，元画像にノイズ(5~20%の一通り)を加えた画像を初期値として想起する実験をせよ．

同条件で画像の種類を6程度まで徐々に増やして想起性能を調べよ．また，画像が2種類と4種類の場合について，ノイズを0％から100％（50%以上の意味は何か？）まで徐々に増やして想起性能を調べよ．

次に，自分なりの疑問を立てて実験的に調べよ．なお，想起性能としては，正解との類似度の全試行平均と，正答率（元画像を完全再現した頻度割合）を用いる．

入力画像はプログラム中でデータ配列として静的に宣言してよいが，公開データを用いてもよい．結果は必ずグラフ等で分かり易く示せ．

上記に規定しない事項については自分で定め，説明せよ．プログラミング経験がない場合，最初にその旨明記した上で，AIやネットを活用したり，詳しい人に教えてもらうなどして（いずれも利用したことを明記），できるだけの努力をしてレポートを作成せよ．