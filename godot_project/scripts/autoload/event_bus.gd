## EventBus - Global signal bus for decoupled communication.
extends Node

# UI Signals
signal show_province_panel(province_id: String)
signal hide_province_panel
signal show_diplomacy_panel
signal hide_diplomacy_panel
signal show_tech_panel
signal hide_tech_panel
signal show_army_panel(army_id: String)
signal hide_army_panel
signal show_character_detail(character_id: String)
signal show_notification(text: String, type: String)

# Game Action Signals
signal action_build(province_id: String, building_type: int)
signal action_recruit_army(province_id: String, soldiers: int)
signal action_move_army(army_id: String, target_province_id: String)
signal action_research(tech_id: String)
signal action_diplomacy(target_faction_id: String, action: String)
signal action_recruit_character(province_id: String)
signal action_assign_general(character_id: String, army_id: String)
signal action_assign_governor(character_id: String, province_id: String)
signal action_end_turn

# Map Signals
signal province_clicked(province_id: String)
signal province_hovered(province_id: String)
signal province_unhovered
signal map_mode_changed(mode: String)

# Multiplayer Signals
signal player_connected(player_id: String, player_name: String)
signal player_disconnected(player_id: String)
signal chat_message_received(sender: String, text: String)
signal server_state_received(state: Dictionary)
