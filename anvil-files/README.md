# Anvil app clone – run from here after authenticating

1. **Authenticate** with Anvil (SSH key in Anvil → Settings → Git, or however you normally connect).

2. **Clone into this folder** (from this directory):
   ```bash
   git clone "ssh://raghu.s%40utexas.edu@anvil.works:2222/ZYMBBJME4FOBYPVB.git" .
   ```
   This puts the app (e.g. `server_code/`, `client_code/`, `anvil.yaml`) directly inside `anvil-files/`.

3. **Apply the Server Module edit:**
   ```bash
   cp ServerModule1_edited.py server_code/ServerModule1.py
   ```

4. **Commit and push:**
   ```bash
   git add server_code/ServerModule1.py
   git commit -m "Server: get_ai_move and check_winner_server provided by Uplink backend"
   git push
   ```

See **../ANVIL_CLONE.md** for more detail.
