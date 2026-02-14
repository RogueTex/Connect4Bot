# ServerModule1: get_ai_move and check_winner_server are provided by the Uplink
# backend (server.py on EC2/Lightsail). When the Uplink is connected, the app
# uses that backend for these callables. Do not define them here.

import anvil.users
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.server
