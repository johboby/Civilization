## MainMenu - Handles the main menu screen logic.
extends Control

@onready var player_name_input: LineEdit = $VBoxContainer/PlayerNameInput
@onready var map_cols_spin: SpinBox = $VBoxContainer/MapColsSpinBox
@onready var map_rows_spin: SpinBox = $VBoxContainer/MapRowsSpinBox
@onready var factions_spin: SpinBox = $VBoxContainer/FactionsSpinBox
@onready var single_player_btn: Button = $VBoxContainer/SinglePlayerBtn
@onready var host_btn: Button = $VBoxContainer/HostGameBtn
@onready var join_btn: Button = $VBoxContainer/JoinGameBtn
@onready var server_url_input: LineEdit = $VBoxContainer/ServerUrlInput
@onready var status_label: Label = $VBoxContainer/StatusLabel

func _ready() -> void:
	single_player_btn.pressed.connect(_on_single_player)
	host_btn.pressed.connect(_on_host_game)
	join_btn.pressed.connect(_on_join_game)
	NetworkManager.connected_to_server.connect(_on_connected)
	NetworkManager.connection_error.connect(_on_connection_error)
	GameData.game_started.connect(_on_game_started)

func _on_single_player() -> void:
	var pname := player_name_input.text.strip_edges()
	if pname.is_empty():
		pname = "Player"

	status_label.text = "Starting game..."
	GameSetup.new_game(
		pname,
		int(factions_spin.value),
		int(map_cols_spin.value),
		int(map_rows_spin.value),
	)

func _on_host_game() -> void:
	status_label.text = "Start the Python server first, then use Join Game."

func _on_join_game() -> void:
	var pname := player_name_input.text.strip_edges()
	if pname.is_empty():
		pname = "Player"

	var url := server_url_input.text.strip_edges()
	if url.is_empty():
		url = "ws://localhost:8000"

	status_label.text = "Connecting..."
	var err := NetworkManager.connect_to_server(url, pname)
	if err != OK:
		status_label.text = "Connection failed!"

func _on_connected() -> void:
	status_label.text = "Connected! Waiting for game..."

func _on_connection_error(msg: String) -> void:
	status_label.text = "Error: %s" % msg

func _on_game_started() -> void:
	get_tree().change_scene_to_file("res://scenes/game_world/game_world.tscn")
