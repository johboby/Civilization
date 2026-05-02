## NetworkManager - Handles WebSocket multiplayer communication.
##
## Connects to the Python game server for online multiplayer.
## Also supports local single-player mode without server.
extends Node

signal connected_to_server
signal disconnected_from_server
signal connection_error(message: String)
signal message_received(data: Dictionary)

var _ws: WebSocketPeer = null
var _connected: bool = false
var _server_url: String = ""
var _player_name: String = "Player"
var _room_id: String = "default"

var is_online_mode: bool = false
var is_host: bool = false

# -----------------------------------------------------------------------
# Connection
# -----------------------------------------------------------------------

func connect_to_server(url: String, player_name: String, room_id: String = "default") -> Error:
	_server_url = url
	_player_name = player_name
	_room_id = room_id

	var full_url := "%s/ws/%s?player_name=%s&player_id=%s" % [
		url, room_id, player_name.uri_encode(), GameData.local_player_id
	]

	_ws = WebSocketPeer.new()
	var err := _ws.connect_to_url(full_url)
	if err != OK:
		connection_error.emit("Failed to connect: %s" % error_string(err))
		return err

	is_online_mode = true
	set_process(true)
	return OK

func disconnect_from_server() -> void:
	if _ws:
		_ws.close()
		_ws = null
	_connected = false
	is_online_mode = false
	set_process(false)
	disconnected_from_server.emit()

func send_message(data: Dictionary) -> void:
	if not _connected or _ws == null:
		return
	var json_str := JSON.stringify(data)
	_ws.send_text(json_str)

# -----------------------------------------------------------------------
# Convenience methods
# -----------------------------------------------------------------------

func send_action(action: Dictionary) -> void:
	send_message({"type": "action", "action": action})

func send_ready() -> void:
	send_message({"type": "ready"})

func send_force_turn() -> void:
	send_message({"type": "force_turn"})

func send_chat(text: String) -> void:
	send_message({"type": "chat", "text": text})

func request_state() -> void:
	send_message({"type": "get_state"})

func select_faction(faction_id: String) -> void:
	send_message({"type": "select_faction", "faction_id": faction_id})

# -----------------------------------------------------------------------
# Process (WebSocket polling)
# -----------------------------------------------------------------------

func _ready() -> void:
	set_process(false)
	# Generate a local player ID
	GameData.local_player_id = _generate_player_id()

func _process(_delta: float) -> void:
	if _ws == null:
		return

	_ws.poll()
	var state := _ws.get_ready_state()

	match state:
		WebSocketPeer.STATE_OPEN:
			if not _connected:
				_connected = true
				connected_to_server.emit()
			while _ws.get_available_packet_count() > 0:
				var packet := _ws.get_packet()
				var text := packet.get_string_from_utf8()
				var json := JSON.new()
				if json.parse(text) == OK:
					_handle_message(json.data)
		WebSocketPeer.STATE_CLOSING:
			pass
		WebSocketPeer.STATE_CLOSED:
			var code := _ws.get_close_code()
			_ws = null
			_connected = false
			is_online_mode = false
			set_process(false)
			disconnected_from_server.emit()

func _handle_message(data: Dictionary) -> void:
	message_received.emit(data)

	var msg_type: String = data.get("type", "")
	match msg_type:
		"connected":
			GameData.local_player_id = data.get("player_id", GameData.local_player_id)
			if data.has("state"):
				_apply_server_state(data["state"])
			if data.has("faction_id"):
				GameData.local_faction_id = data["faction_id"]

		"game_started":
			if data.has("state"):
				_apply_server_state(data["state"])
			GameData.game_started.emit()

		"turn_processed":
			if data.has("state"):
				_apply_server_state(data["state"])
			var events: Array = data.get("events", [])
			GameData.turn_processed.emit(data.get("turn", 0), events)

		"state_update":
			if data.has("state"):
				_apply_server_state(data["state"])

		"faction_selected":
			GameData.local_faction_id = data.get("faction_id", "")
			GameData.faction_changed.emit(GameData.local_faction_id)

		"chat":
			EventBus.chat_message_received.emit(
				data.get("name", "Unknown"),
				data.get("text", "")
			)

		"player_joined":
			EventBus.player_connected.emit(
				data.get("player_id", ""),
				data.get("name", "Unknown")
			)

		"player_left":
			EventBus.player_disconnected.emit(data.get("player_id", ""))

func _apply_server_state(state: Dictionary) -> void:
	GameData.turn = state.get("turn", 0)
	GameData.season = state.get("season", 0)
	GameData.year = state.get("year", 190)
	if state.has("provinces"):
		GameData.provinces = state["provinces"]
	if state.has("factions"):
		GameData.factions = state["factions"]
	if state.has("characters"):
		GameData.characters = state["characters"]
	if state.has("armies"):
		GameData.armies = state["armies"]
	if state.has("diplomacy"):
		GameData.diplomacy = state["diplomacy"]
	if state.has("events"):
		for evt in state["events"]:
			GameData.event_log.append(evt)
	GameData.game_active = true
	EventBus.server_state_received.emit(state)

func _generate_player_id() -> String:
	var chars := "abcdef0123456789"
	var result := ""
	for i in 16:
		result += chars[randi() % chars.length()]
	return result
