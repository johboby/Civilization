## SidePanel - Manages the right-side information panel with tabs.
extends PanelContainer

enum Tab { PROVINCE, ARMY, OFFICERS, DIPLOMACY, TECH }

@onready var province_btn: Button = $VBox/TabBar/ProvinceTabBtn
@onready var army_btn: Button = $VBox/TabBar/ArmyTabBtn
@onready var officer_btn: Button = $VBox/TabBar/OfficerTabBtn
@onready var diplo_btn: Button = $VBox/TabBar/DiploTabBtn
@onready var tech_btn: Button = $VBox/TabBar/TechTabBtn
@onready var content_vbox: VBoxContainer = $VBox/ScrollContainer/ContentVBox

var _current_tab: Tab = Tab.PROVINCE
var _tab_buttons: Array[Button] = []

func _ready() -> void:
	_tab_buttons = [province_btn, army_btn, officer_btn, diplo_btn, tech_btn]

	province_btn.pressed.connect(func(): _switch_tab(Tab.PROVINCE))
	army_btn.pressed.connect(func(): _switch_tab(Tab.ARMY))
	officer_btn.pressed.connect(func(): _switch_tab(Tab.OFFICERS))
	diplo_btn.pressed.connect(func(): _switch_tab(Tab.DIPLOMACY))
	tech_btn.pressed.connect(func(): _switch_tab(Tab.TECH))

	EventBus.province_clicked.connect(_on_province_clicked)
	GameData.turn_processed.connect(func(_t, _e): _refresh())
	GameData.game_started.connect(_refresh)
	EventBus.server_state_received.connect(func(_s): _refresh())

	_refresh()

func _switch_tab(tab: Tab) -> void:
	_current_tab = tab
	for i in _tab_buttons.size():
		_tab_buttons[i].button_pressed = (i == tab)
	_refresh()

func _on_province_clicked(province_id: String) -> void:
	_current_tab = Tab.PROVINCE
	for i in _tab_buttons.size():
		_tab_buttons[i].button_pressed = (i == 0)
	_refresh()

func _refresh() -> void:
	# Clear content
	for child in content_vbox.get_children():
		child.queue_free()

	match _current_tab:
		Tab.PROVINCE: _render_province()
		Tab.ARMY: _render_armies()
		Tab.OFFICERS: _render_officers()
		Tab.DIPLOMACY: _render_diplomacy()
		Tab.TECH: _render_tech()

# -----------------------------------------------------------------------
# Province Tab
# -----------------------------------------------------------------------

func _render_province() -> void:
	var province := GameData.provinces.get(GameData.selected_province_id, {})
	if province.is_empty():
		_add_label("Select a province on the map")
		return

	_add_header(province["name"])

	var owner_name := "Uncontrolled"
	if province["owner_faction_id"] != "":
		var faction := GameData.factions.get(province["owner_faction_id"], {})
		owner_name = faction.get("name", "Unknown")
	_add_row("Owner", owner_name)
	_add_row("Terrain", GameData.TERRAIN_NAMES.get(province["terrain"], "?"))
	_add_row("Population", str(province["population"]))
	_add_row("Food", str(int(province["food"])))
	_add_row("Gold", str(int(province["gold"])))
	_add_row("Manpower", str(province["manpower"]))
	_add_row("Development", "%.2f" % province["development"])
	_add_row("Fortification", str(province["fortification"]))
	_add_row("Unrest", "%.1f" % province["unrest"])

	# Buildings
	if province["buildings"].size() > 0:
		_add_header("Buildings")
		for b in province["buildings"]:
			var bname: String = GameData.BUILDING_NAMES.get(b["type"], "?")
			_add_row(bname, "Lv %d" % b["level"])

	# Actions (only for own provinces)
	if province["owner_faction_id"] == GameData.local_faction_id:
		_add_header("Actions")
		var actions_box := HBoxContainer.new()
		actions_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL

		var build_types := [
			[GameData.BuildingType.FARM, "Farm"],
			[GameData.BuildingType.MARKET, "Market"],
			[GameData.BuildingType.BARRACKS, "Barracks"],
			[GameData.BuildingType.WALL, "Wall"],
			[GameData.BuildingType.ACADEMY, "Academy"],
		]

		var grid := GridContainer.new()
		grid.columns = 2
		grid.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		for bt in build_types:
			var btn := Button.new()
			btn.text = "Build %s (%d)" % [bt[1], GameData.BUILDING_COSTS.get(bt[0], 0)]
			btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			var btype: int = bt[0]
			var pid: String = province["id"]
			btn.pressed.connect(func(): EventBus.action_build.emit(pid, btype))
			grid.add_child(btn)

		content_vbox.add_child(grid)

		var recruit_btn := Button.new()
		recruit_btn.text = "Recruit Army (1000 soldiers)"
		recruit_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		var pid2: String = province["id"]
		recruit_btn.pressed.connect(func(): EventBus.action_recruit_army.emit(pid2, 1000))
		content_vbox.add_child(recruit_btn)

		var char_btn := Button.new()
		char_btn.text = "Recruit Officer (100 gold)"
		char_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		var pid3: String = province["id"]
		char_btn.pressed.connect(func(): EventBus.action_recruit_character.emit(pid3))
		content_vbox.add_child(char_btn)

# -----------------------------------------------------------------------
# Army Tab
# -----------------------------------------------------------------------

func _render_armies() -> void:
	_add_header("Your Armies")
	var armies := GameData.get_faction_armies(GameData.local_faction_id)
	if armies.is_empty():
		_add_label("No armies")
		return

	for army in armies:
		var panel := PanelContainer.new()
		var vbox := VBoxContainer.new()
		panel.add_child(vbox)

		var province := GameData.provinces.get(army["province_id"], {})
		var pname: String = province.get("name", "Unknown")
		var general := GameData.characters.get(army.get("general_id", ""), {})
		var gname: String = general.get("name", "No general")

		_add_label_to(vbox, "%d soldiers" % army["soldiers"])
		_add_label_to(vbox, "At: %s | Status: %s" % [pname,
			["Idle", "Marching", "Besieging", "Defending", "Retreating"][army["status"]]])
		_add_label_to(vbox, "General: %s" % gname)
		_add_label_to(vbox, "Inf: %d | Cav: %d | Arc: %d | Morale: %d" % [
			army["infantry"], army["cavalry"], army["archers"], int(army["morale"])])

		# Movement buttons
		if army["status"] == GameData.ArmyStatus.IDLE and not province.is_empty():
			var move_box := HBoxContainer.new()
			for adj_id in province.get("adjacent_ids", []).slice(0, 4):
				var adj := GameData.provinces.get(adj_id, {})
				if not adj.is_empty():
					var btn := Button.new()
					btn.text = adj["name"].substr(0, 6)
					var aid: String = army["id"]
					var tid: String = adj_id
					btn.pressed.connect(func(): EventBus.action_move_army.emit(aid, tid))
					move_box.add_child(btn)
			vbox.add_child(move_box)

		content_vbox.add_child(panel)

# -----------------------------------------------------------------------
# Officers Tab
# -----------------------------------------------------------------------

func _render_officers() -> void:
	_add_header("Your Officers")
	var chars := GameData.get_faction_characters(GameData.local_faction_id)
	if chars.is_empty():
		_add_label("No officers")
		return

	for ch in chars:
		var panel := PanelContainer.new()
		var vbox := VBoxContainer.new()
		panel.add_child(vbox)

		var role_names := ["Ruler", "General", "Strategist", "Governor", "Diplomat", "Spy", "Free"]
		var role_name: String = role_names[ch["role"]] if ch["role"] < role_names.size() else "?"

		_add_label_to(vbox, "%s - %s (Lv %d)" % [ch["name"], role_name, ch["level"]])
		_add_label_to(vbox, "Age: %d | Loyalty: %d" % [ch["age"], ch["loyalty"]])
		_add_label_to(vbox, "CMD:%d FOR:%d INT:%d POL:%d CHA:%d" % [
			ch["command"], ch["force"], ch["intelligence"], ch["politics"], ch["charisma"]])

		content_vbox.add_child(panel)

# -----------------------------------------------------------------------
# Diplomacy Tab
# -----------------------------------------------------------------------

func _render_diplomacy() -> void:
	_add_header("Diplomacy")
	var status_names := ["Neutral", "Ally", "War", "Vassal", "Truce", "Trade"]

	for rel in GameData.diplomacy.values():
		var other_id := ""
		if rel["faction_a_id"] == GameData.local_faction_id:
			other_id = rel["faction_b_id"]
		elif rel["faction_b_id"] == GameData.local_faction_id:
			other_id = rel["faction_a_id"]
		else:
			continue

		var other := GameData.factions.get(other_id, {})
		if other.is_empty():
			continue

		var panel := PanelContainer.new()
		var vbox := VBoxContainer.new()
		panel.add_child(vbox)

		var status_name: String = status_names[rel["status"]] if rel["status"] < status_names.size() else "?"
		_add_label_to(vbox, "%s - %s" % [other["name"], status_name])
		_add_label_to(vbox, "Opinion: %d | Trade: %d" % [rel["opinion"], int(rel["trade_value"])])

		var btn_box := HBoxContainer.new()
		var actions := ["ally", "trade", "war", "truce"]
		var action_labels := ["Ally", "Trade", "War", "Truce"]
		for i in actions.size():
			var btn := Button.new()
			btn.text = action_labels[i]
			var oid: String = other_id
			var act: String = actions[i]
			btn.pressed.connect(func(): EventBus.action_diplomacy.emit(oid, act))
			btn_box.add_child(btn)
		vbox.add_child(btn_box)
		content_vbox.add_child(panel)

# -----------------------------------------------------------------------
# Tech Tab
# -----------------------------------------------------------------------

func _render_tech() -> void:
	_add_header("Technology")
	var faction := GameData.factions.get(GameData.local_faction_id, {})
	if faction.is_empty():
		return

	_add_row("Military", "Lv %d" % faction["tech_levels"].get("military", 1))
	_add_row("Economy", "Lv %d" % faction["tech_levels"].get("economy", 1))
	_add_row("Culture", "Lv %d" % faction["tech_levels"].get("culture", 1))
	_add_row("Diplomacy", "Lv %d" % faction["tech_levels"].get("diplomacy", 1))
	_add_row("Research Points", str(int(faction["research_points"])))

	if faction["current_research"] != "":
		_add_row("Researching", faction["current_research"])

	_add_header("Available Research")
	var available := TechTree.get_available(faction["researched_techs"])
	for tech in available:
		var panel := PanelContainer.new()
		var vbox := VBoxContainer.new()
		panel.add_child(vbox)

		var cat_names := ["Military", "Economy", "Culture", "Diplomacy"]
		var cat_name: String = cat_names[tech["category"]] if tech["category"] < cat_names.size() else "?"

		_add_label_to(vbox, "%s (Lv %d) - Cost: %d" % [tech["name"], tech["level"], tech["cost"]])
		_add_label_to(vbox, "[%s] %s" % [cat_name, tech["description"]])

		var btn := Button.new()
		btn.text = "Research"
		if faction["current_research"] == tech["id"]:
			btn.text = "Researching..."
			btn.disabled = true
		var tid: String = tech["id"]
		btn.pressed.connect(func(): EventBus.action_research.emit(tid))
		vbox.add_child(btn)

		content_vbox.add_child(panel)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

func _add_header(text: String) -> void:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 14)
	content_vbox.add_child(label)
	var sep := HSeparator.new()
	content_vbox.add_child(sep)

func _add_label(text: String) -> void:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 12)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	content_vbox.add_child(label)

func _add_row(key: String, value: String) -> void:
	var hbox := HBoxContainer.new()
	var key_label := Label.new()
	key_label.text = key
	key_label.add_theme_font_size_override("font_size", 11)
	key_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	var val_label := Label.new()
	val_label.text = value
	val_label.add_theme_font_size_override("font_size", 11)
	val_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	hbox.add_child(key_label)
	hbox.add_child(val_label)
	content_vbox.add_child(hbox)

func _add_label_to(parent: Node, text: String) -> void:
	var label := Label.new()
	label.text = text
	label.add_theme_font_size_override("font_size", 11)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	parent.add_child(label)
