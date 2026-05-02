## GameHUD - Top bar and bottom panel controller.
extends Control

@onready var faction_label: Label = $TopBar/HBox/FactionLabel
@onready var turn_label: Label = $TopBar/HBox/TurnLabel
@onready var year_label: Label = $TopBar/HBox/YearLabel
@onready var season_label: Label = $TopBar/HBox/SeasonLabel
@onready var gold_label: Label = $TopBar/HBox/GoldLabel
@onready var food_label: Label = $TopBar/HBox/FoodLabel
@onready var manpower_label: Label = $TopBar/HBox/ManpowerLabel
@onready var end_turn_btn: Button = $TopBar/HBox/EndTurnBtn
@onready var save_btn: Button = $TopBar/HBox/SaveBtn
@onready var event_list: VBoxContainer = $BottomPanel/HSplit/EventLog/EventScroll/EventList
@onready var event_scroll: ScrollContainer = $BottomPanel/HSplit/EventLog/EventScroll
@onready var chat_list: VBoxContainer = $BottomPanel/HSplit/ChatPanel/ChatScroll/ChatList
@onready var chat_scroll: ScrollContainer = $BottomPanel/HSplit/ChatPanel/ChatScroll
@onready var chat_input: LineEdit = $BottomPanel/HSplit/ChatPanel/ChatInputRow/ChatInput
@onready var chat_send_btn: Button = $BottomPanel/HSplit/ChatPanel/ChatInputRow/ChatSendBtn

func _ready() -> void:
	end_turn_btn.pressed.connect(func(): EventBus.action_end_turn.emit())
	save_btn.pressed.connect(_on_save)
	chat_send_btn.pressed.connect(_send_chat)
	chat_input.text_submitted.connect(func(_t): _send_chat())

	GameData.game_started.connect(_update_hud)
	GameData.turn_processed.connect(_on_turn_processed)
	GameData.faction_changed.connect(func(_f): _update_hud())
	EventBus.server_state_received.connect(func(_s): _update_hud())
	EventBus.chat_message_received.connect(_on_chat_received)

	_update_hud()

func _on_turn_processed(_turn: int, events: Array) -> void:
	_update_hud()
	for evt in events:
		_add_event_entry(evt)

func _update_hud() -> void:
	var faction := GameData.factions.get(GameData.local_faction_id, {})

	if not faction.is_empty():
		faction_label.text = "Faction: %s" % faction["name"]
		gold_label.text = "Gold: %d" % int(faction["gold"])
		food_label.text = "Food: %d" % int(faction["food"])
		manpower_label.text = "Manpower: %d" % faction["manpower"]
	else:
		faction_label.text = "Faction: -"
		gold_label.text = "Gold: 0"
		food_label.text = "Food: 0"
		manpower_label.text = "Manpower: 0"

	turn_label.text = "Turn: %d" % GameData.turn
	year_label.text = "Year: %d" % GameData.year
	season_label.text = "Season: %s" % GameData.SEASON_NAMES.get(GameData.season, "?")

func _add_event_entry(evt: Dictionary) -> void:
	var label := Label.new()
	label.text = "[%d] %s" % [evt.get("turn", 0), evt.get("description", "")]
	label.add_theme_font_size_override("font_size", 11)
	event_list.add_child(label)
	# Auto-scroll
	await get_tree().process_frame
	event_scroll.scroll_vertical = event_scroll.get_v_scroll_bar().max_value

func _send_chat() -> void:
	var text := chat_input.text.strip_edges()
	if text.is_empty():
		return
	chat_input.text = ""
	if NetworkManager.is_online_mode:
		NetworkManager.send_chat(text)
	else:
		_on_chat_received("You", text)

func _on_chat_received(sender: String, text: String) -> void:
	var label := Label.new()
	label.text = "%s: %s" % [sender, text]
	label.add_theme_font_size_override("font_size", 11)
	chat_list.add_child(label)
	await get_tree().process_frame
	chat_scroll.scroll_vertical = chat_scroll.get_v_scroll_bar().max_value

func _on_save() -> void:
	var path := "user://save_%d.json" % GameData.turn
	var err := GameData.save_game(path)
	if err == OK:
		EventBus.show_notification.emit("Game saved!", "success")
	else:
		EventBus.show_notification.emit("Save failed!", "error")
