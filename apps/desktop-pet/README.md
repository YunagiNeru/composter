# Desktop Pet

Windows 11向けの猫デスクトップペットです。PC状態の表示、ふれあい、LayaのNPU行動判断を試せます。

## 起動

作業ディレクトリを `C:\work\composter\apps\desktop-pet` にし、専用UI仮想環境で起動します。

```powershell
Set-Location C:\work\composter\apps\desktop-pet
& ..\..\som-probe\.local\venv-desktop-pet\Scripts\python.exe -m desktop_pet
```

猫が画面に表示されます。AIを有効にしている場合は、別プロセスで固定LayaモデルをNPUへ読み込みます。3720／PLUGINを確認できない場合はCPUへ切り替えず、AI利用不可を表示します。

## 操作

- 猫を右クリックして、状態表示、設定、ふれあい、非表示、終了を選びます。
- 猫をクリックすると撫でます。右クリックメニューから餌や遊びを選べます。
- 移動モードではドラッグで位置を変え、`Esc`で解除します。
- システムトレイから表示の復帰、AI状態の確認、終了ができます。

## 架空メトリクスでの確認

```powershell
Set-Location C:\work\composter\apps\desktop-pet
& ..\..\som-probe\.local\venv-desktop-pet\Scripts\python.exe -m desktop_pet --demo-metrics
```

6秒ごとに固定の架空PC状態へ切り替わります。実測値ではなく、設定も保存しません。

## 検証

```powershell
Set-Location C:\work\composter\apps\desktop-pet
& ..\..\som-probe\.local\venv-desktop-pet\Scripts\python.exe -m unittest discover -s tests -v
```

猫のPNGは `scripts\generate_sprites.py` から再生成できます。アプリ設計、設定、制約、検証結果は [DESIGN.html](DESIGN.html) に記録しています。
