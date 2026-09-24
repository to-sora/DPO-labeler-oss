Imported image review / 外部圖片標註 / 外部图片标注
================================================

English
-------
Run `dpo_labeler/start.sh` from this checkout. Open
`https://127.0.0.1:8787`, accept the self-signed certificate, and select
**External images**. Enter a reviewer name; no invite token is needed.

Upload or paste [example YAML](examples/imported.yaml). `image_dir` is an
absolute directory on the backend PC. Image paths are relative to it.
Existing absolute or relative prompt files override inline prompts;
missing prompt files use `prompt`. Filter tasks by task, character, or path.
Every import creates a separate task. Reimport after changing source images.

Choose A or B in every open dimension. Transitive preferences are locked.
Each accepted comparison creates one event per dimension with distinct
`dataset_id` values and a shared `comparison_id`. Inferred events are marked.
Retries preserve events and exposure counts.

Download any dimension or all dimensions at any time. ZIP contains source
references, prompt snapshots, events, DPO pairs, and rankings. DPO pairs keep
separate chosen/rejected prompts; only human choices enter DPO files.
Five disjoint groups and cumulative
0.2/0.4/0.6/0.8 lists include rank intervals and certificate flags.
Uncertified output is an estimate. Certificates use floor(N/20) rank tolerance,
conditional on consistent accepted judgments. The MVP uses shared cached
merge comparisons and exposure priority; it claims no optimal complexity.

Each task's SQLite cache and exports live together under
`--state-dir/imported/<task-id>` (default `output/labeler`). Delete that folder
to remove a task. `--stop` stops launcher-owned services; `--force` frees the
selected port. Port policy is in `config/port-config.yaml`; default is public,
IPv4, without a whitelist. TLS files are in ignored `dpo_labeler/data/tls`.

繁體中文
--------
執行 `dpo_labeler/start.sh`，開啟 HTTPS 網址並信任自簽憑證。
選擇 External images，輸入名稱、匯入 YAML；可按任務、角色或路徑篩選。
圖片目錄位於後端電腦；提示詞檔案優先，缺失時使用行內提示詞。
每維必選 A 或 B，已推論結果鎖定。各維事件獨立且共用比較識別碼。
隨時可匯出逐維 DPO、事件及四個分界；未驗證排名標示為估計。
ZIP 保留來源路徑，圖片不複製。刪除任務輸出資料夾即可清除快取。

简体中文
--------
运行 `dpo_labeler/start.sh`，打开 HTTPS 地址并信任自签证书。
选择 External images，输入名称、导入 YAML；可按任务、角色或路径筛选。
图片目录位于后端电脑；提示词文件优先，缺失时使用行内提示词。
每维必选 A 或 B，已推断结果锁定。各维事件独立且共享比较标识。
随时可导出逐维 DPO、事件及四个分界；未验证排名标记为估计。
ZIP 保留来源路径，图片不复制。删除任务输出文件夹即可清理缓存。
