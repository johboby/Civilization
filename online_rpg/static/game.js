/**
 * Civilization Online RPG - Game Client
 * Handles WebSocket communication, map rendering, and UI interaction.
 */

// ============================================================
// Global State
// ============================================================

const G = {
    ws: null,
    playerId: null,
    playerName: '',
    roomId: 'default',
    factionId: null,
    state: null,       // Full game state
    selectedProvince: null,
    selectedArmy: null,
    mapOffset: { x: 0, y: 0 },
    mapZoom: 1.0,
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    hoveredProvince: null,
    tooltip: null,
};

// Hex geometry constants
const HEX = {
    size: 40,
    width: 0,
    height: 0,
};
HEX.width = HEX.size * 2;
HEX.height = Math.sqrt(3) * HEX.size;

// Terrain colors
const TERRAIN_COLORS = {
    plains: '#4a7c3f',
    mountains: '#8B7355',
    forest: '#2d5a1e',
    desert: '#c9a96e',
    river: '#3a7ca5',
    coast: '#5b9bd5',
    marsh: '#4a6741',
};

// ============================================================
// DOM References
// ============================================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const DOM = {
    loginScreen: $('#login-screen'),
    lobbyScreen: $('#lobby-screen'),
    gameScreen: $('#game-screen'),
    playerName: $('#player-name'),
    serverUrl: $('#server-url'),
    roomId: $('#room-id'),
    btnConnect: $('#btn-connect'),
    connStatus: $('#connection-status'),
    playerList: $('#player-list'),
    btnStartGame: $('#btn-start-game'),
    factionSelectList: $('#faction-select-list'),
    canvas: $('#game-map'),
    infoFaction: $('#info-faction'),
    infoTurn: $('#info-turn'),
    infoYear: $('#info-year'),
    infoSeason: $('#info-season'),
    resGold: $('#res-gold'),
    resFood: $('#res-food'),
    resManpower: $('#res-manpower'),
    btnEndTurn: $('#btn-end-turn'),
    btnSave: $('#btn-save'),
    provinceName: $('#province-name'),
    provinceDetails: $('#province-details'),
    provinceActions: $('#province-actions'),
    armyList: $('#army-list'),
    characterList: $('#character-list'),
    diplomacyList: $('#diplomacy-list'),
    techInfo: $('#tech-info'),
    techAvailable: $('#tech-available'),
    eventList: $('#event-list'),
    chatMessages: $('#chat-messages'),
    chatInput: $('#chat-input'),
    btnChatSend: $('#btn-chat-send'),
};

// ============================================================
// Connection
// ============================================================

function connect() {
    G.playerName = DOM.playerName.value.trim() || 'Player';
    G.roomId = DOM.roomId.value.trim() || 'default';

    let serverBase = DOM.serverUrl.value.trim();
    if (!serverBase) {
        const loc = window.location;
        const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
        serverBase = `${wsProto}//${loc.host}`;
    }
    // Ensure ws:// prefix
    if (!serverBase.startsWith('ws://') && !serverBase.startsWith('wss://')) {
        serverBase = 'ws://' + serverBase;
    }

    const url = `${serverBase}/ws/${G.roomId}?player_name=${encodeURIComponent(G.playerName)}&player_id=${G.playerId || ''}`;

    DOM.connStatus.textContent = 'Connecting...';
    DOM.connStatus.style.color = '#f39c12';

    G.ws = new WebSocket(url);

    G.ws.onopen = () => {
        DOM.connStatus.textContent = 'Connected!';
        DOM.connStatus.style.color = '#2ecc71';
    };

    G.ws.onmessage = (evt) => {
        try {
            const data = JSON.parse(evt.data);
            handleMessage(data);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };

    G.ws.onclose = () => {
        DOM.connStatus.textContent = 'Disconnected';
        DOM.connStatus.style.color = '#e74c3c';
    };

    G.ws.onerror = () => {
        DOM.connStatus.textContent = 'Connection error';
        DOM.connStatus.style.color = '#e74c3c';
    };
}

function sendWS(data) {
    if (G.ws && G.ws.readyState === WebSocket.OPEN) {
        G.ws.send(JSON.stringify(data));
    }
}

// ============================================================
// Message Handler
// ============================================================

function handleMessage(data) {
    switch (data.type) {
        case 'connected':
            G.playerId = data.player_id;
            G.roomId = data.room_id;
            if (data.state) {
                G.state = data.state;
                G.factionId = data.faction_id || null;
                showScreen('game');
                updateAll();
            } else {
                showScreen('lobby');
                updatePlayerList(data.players || []);
            }
            break;

        case 'player_joined':
        case 'player_left':
            // Refresh player list if in lobby
            break;

        case 'game_started':
            G.state = data.state;
            showScreen('game');
            updateAll();
            break;

        case 'turn_processed':
            G.state = data.state;
            addEvents(data.events || []);
            updateAll();
            break;

        case 'state_update':
            G.state = data.state;
            updateAll();
            break;

        case 'faction_selected':
            G.factionId = data.faction_id;
            updateAll();
            break;

        case 'action_result':
            if (!data.ok) {
                addChatMessage('System', data.error || 'Action failed', '#e74c3c');
            }
            break;

        case 'player_ready':
            addChatMessage('System', `Player is ready. All ready: ${data.all_ready}`, '#3498db');
            break;

        case 'chat':
            addChatMessage(data.name, data.text, null, data.faction_id);
            break;

        default:
            console.log('Unknown message type:', data.type);
    }
}

// ============================================================
// Screen Management
// ============================================================

function showScreen(name) {
    DOM.loginScreen.classList.add('hidden');
    DOM.lobbyScreen.classList.add('hidden');
    DOM.gameScreen.classList.add('hidden');

    if (name === 'login') DOM.loginScreen.classList.remove('hidden');
    else if (name === 'lobby') DOM.lobbyScreen.classList.remove('hidden');
    else if (name === 'game') {
        DOM.gameScreen.classList.remove('hidden');
        resizeCanvas();
        renderMap();
    }
}

// ============================================================
// Lobby
// ============================================================

function updatePlayerList(players) {
    DOM.playerList.innerHTML = '';
    players.forEach(p => {
        const li = document.createElement('li');
        li.textContent = `${p.name} ${p.faction_id ? '(in game)' : ''}`;
        DOM.playerList.appendChild(li);
    });
}

async function startGame() {
    const mapCols = parseInt($('#map-cols').value) || 10;
    const mapRows = parseInt($('#map-rows').value) || 8;
    const numFactions = parseInt($('#num-factions').value) || 4;

    const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];
    const aiNames = ['Wei Empire', 'Shu Kingdom', 'Wu Dynasty', 'Jin Alliance', 'Yan State', 'Chu Domain', 'Qi Realm', 'Qin Authority'];

    const factions = [];
    // First faction is always the player
    factions.push({
        name: `${G.playerName}'s Kingdom`,
        color: colors[0],
        player_id: G.playerId,
    });
    for (let i = 1; i < numFactions; i++) {
        factions.push({
            name: aiNames[i % aiNames.length],
            color: colors[i % colors.length],
            player_id: null,
        });
    }

    try {
        const resp = await fetch(`/api/rooms/${G.roomId}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ factions, map_cols: mapCols, map_rows: mapRows }),
        });
        const result = await resp.json();
        if (!result.ok) {
            alert(result.error || 'Failed to start game');
        }
    } catch (e) {
        console.error('Start game error:', e);
    }
}

// ============================================================
// UI Updates
// ============================================================

function updateAll() {
    if (!G.state) return;
    updateTopBar();
    renderMap();
    updateSidePanel();
    updateArmyList();
    updateCharacterList();
    updateDiplomacy();
    updateTechPanel();
}

function updateTopBar() {
    const s = G.state;
    const faction = G.factionId ? s.factions[G.factionId] : null;

    DOM.infoTurn.textContent = `Turn: ${s.turn}`;
    DOM.infoYear.textContent = `Year: ${s.year}`;
    DOM.infoSeason.textContent = `Season: ${capitalize(s.season)}`;

    if (faction) {
        DOM.infoFaction.textContent = `Faction: ${faction.name}`;
        DOM.infoFaction.style.color = faction.color;
        DOM.resGold.textContent = `Gold: ${Math.floor(faction.gold)}`;
        DOM.resFood.textContent = `Food: ${Math.floor(faction.food)}`;
        DOM.resManpower.textContent = `Manpower: ${faction.manpower}`;
    }
}

function updateSidePanel() {
    if (!G.selectedProvince) {
        DOM.provinceName.textContent = 'Select a province';
        DOM.provinceDetails.innerHTML = '';
        DOM.provinceActions.innerHTML = '';
        return;
    }

    const p = G.state.provinces[G.selectedProvince];
    if (!p) return;

    DOM.provinceName.textContent = p.name;

    const owner = p.owner_faction_id ? G.state.factions[p.owner_faction_id] : null;
    const ownerName = owner ? owner.name : 'Uncontrolled';
    const isOwn = p.owner_faction_id === G.factionId;

    let html = `
        <div class="detail-row"><span class="detail-label">Owner</span><span class="detail-value" style="color:${owner ? owner.color : '#888'}">${ownerName}</span></div>
        <div class="detail-row"><span class="detail-label">Terrain</span><span class="detail-value">${capitalize(p.terrain)}</span></div>
        <div class="detail-row"><span class="detail-label">Population</span><span class="detail-value">${p.population.toLocaleString()}</span></div>
        <div class="detail-row"><span class="detail-label">Food</span><span class="detail-value">${Math.floor(p.food)}</span></div>
        <div class="detail-row"><span class="detail-label">Gold</span><span class="detail-value">${Math.floor(p.gold)}</span></div>
        <div class="detail-row"><span class="detail-label">Manpower</span><span class="detail-value">${p.manpower}</span></div>
        <div class="detail-row"><span class="detail-label">Development</span><span class="detail-value">${p.development.toFixed(2)}</span></div>
        <div class="detail-row"><span class="detail-label">Fortification</span><span class="detail-value">${p.fortification}</span></div>
        <div class="detail-row"><span class="detail-label">Unrest</span><span class="detail-value">${p.unrest.toFixed(1)}</span></div>
    `;

    if (p.buildings && p.buildings.length > 0) {
        html += '<div style="margin-top:8px;font-size:13px;color:#f5c518">Buildings:</div>';
        p.buildings.forEach(b => {
            html += `<div class="detail-row"><span class="detail-label">${capitalize(b.type)}</span><span class="detail-value">Lv ${b.level}</span></div>`;
        });
    }

    DOM.provinceDetails.innerHTML = html;

    // Actions
    let actHtml = '';
    if (isOwn) {
        actHtml = `
            <div class="action-btn-group">
                <button class="btn btn-small" onclick="buildInProvince('farm')">Build Farm</button>
                <button class="btn btn-small" onclick="buildInProvince('market')">Build Market</button>
                <button class="btn btn-small" onclick="buildInProvince('barracks')">Build Barracks</button>
                <button class="btn btn-small" onclick="buildInProvince('wall')">Build Wall</button>
                <button class="btn btn-small" onclick="buildInProvince('academy')">Build Academy</button>
                <button class="btn btn-small" onclick="buildInProvince('workshop')">Build Workshop</button>
                <button class="btn btn-small" onclick="recruitArmy()">Recruit Army</button>
                <button class="btn btn-small" onclick="recruitCharacter()">Recruit Officer</button>
            </div>
        `;
    }
    DOM.provinceActions.innerHTML = actHtml;
}

function updateArmyList() {
    if (!G.state || !G.factionId) { DOM.armyList.innerHTML = ''; return; }

    const armies = Object.values(G.state.armies).filter(a => a.faction_id === G.factionId);
    let html = '';
    armies.forEach(a => {
        const province = G.state.provinces[a.province_id];
        const general = a.general_id ? G.state.characters[a.general_id] : null;
        const pname = province ? province.name : 'Unknown';
        const gname = general ? general.name : 'No general';

        html += `
            <div class="army-card" onclick="selectArmy('${a.id}')">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span class="army-soldiers">${a.soldiers.toLocaleString()}</span>
                    <span style="font-size:11px;color:#a0a0b0">${capitalize(a.status)}</span>
                </div>
                <div style="font-size:12px;margin-top:4px">
                    <span style="color:#a0a0b0">At:</span> ${pname}
                </div>
                <div style="font-size:12px">
                    <span style="color:#a0a0b0">General:</span> ${gname}
                </div>
                <div style="font-size:11px;color:#a0a0b0;margin-top:2px">
                    Inf: ${a.infantry} | Cav: ${a.cavalry} | Arc: ${a.archers} | Morale: ${Math.floor(a.morale)}
                </div>
                ${a.status === 'idle' && G.selectedProvince && province ?
                    `<div class="action-btn-group" style="margin-top:6px">
                        ${getAdjacentMoveButtons(a, province)}
                    </div>` : ''}
            </div>
        `;
    });
    DOM.armyList.innerHTML = html || '<div style="color:#a0a0b0;font-size:13px">No armies</div>';
}

function getAdjacentMoveButtons(army, currentProvince) {
    if (!currentProvince || !currentProvince.adjacent_province_ids) return '';
    let btns = '';
    currentProvince.adjacent_province_ids.slice(0, 6).forEach(pid => {
        const target = G.state.provinces[pid];
        if (target) {
            btns += `<button class="btn btn-small" onclick="moveArmy('${army.id}','${pid}')">${target.name}</button>`;
        }
    });
    return btns;
}

// Portrait cache
const _portraitCache = {};

function getPortraitUrl(name, role, stats) {
    const key = `${name}_${role}`;
    if (_portraitCache[key]) return _portraitCache[key];
    const url = `/api/portrait/${encodeURIComponent(name)}?role=${role}&command=${stats.command}&force=${stats.force}&intelligence=${stats.intelligence}&politics=${stats.politics}&charisma=${stats.charisma}`;
    _portraitCache[key] = url;
    return url;
}

function updateCharacterList() {
    if (!G.state || !G.factionId) { DOM.characterList.innerHTML = ''; return; }

    const chars = Object.values(G.state.characters)
        .filter(c => c.faction_id === G.factionId && c.alive);

    let html = '';
    chars.forEach(c => {
        const s = c.stats;
        const portraitUrl = getPortraitUrl(c.name, c.role, s);
        const roleColors = {ruler:'#ffd700',general:'#e74c3c',strategist:'#3498db',governor:'#2ecc71',diplomat:'#9b59b6',spy:'#555',free:'#888'};
        const roleColor = roleColors[c.role] || '#888';
        const totalPower = s.command + s.force + s.intelligence + s.politics + s.charisma;
        html += `
            <div class="char-card">
                <div style="display:flex;gap:10px;align-items:flex-start">
                    <img src="${portraitUrl}" width="60" height="75" style="border-radius:4px;flex-shrink:0" alt="${c.name}">
                    <div style="flex:1;min-width:0">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <span class="char-name">${c.name}</span>
                            <span class="char-role" style="color:${roleColor}">${c.role}</span>
                        </div>
                        <div style="font-size:11px;color:#8890a8;margin:2px 0">Lv ${c.level} | Age ${c.age} | Power ${totalPower}</div>
                        <div style="display:flex;align-items:center;gap:4px;margin:2px 0">
                            <span style="font-size:10px;color:#8890a8">Loyalty:</span>
                            <div class="progress-bar" style="flex:1;height:4px">
                                <div class="progress-bar-fill ${s.loyalty > 50 ? 'green' : 'red'}" style="width:${s.loyalty}%"></div>
                            </div>
                            <span style="font-size:10px;color:${s.loyalty > 50 ? '#2ecc71' : '#e74c3c'}">${s.loyalty}</span>
                        </div>
                        <div class="char-stats">
                            <div class="char-stat"><span>CMD</span><span>${s.command}</span></div>
                            <div class="char-stat"><span>FOR</span><span>${s.force}</span></div>
                            <div class="char-stat"><span>INT</span><span>${s.intelligence}</span></div>
                            <div class="char-stat"><span>POL</span><span>${s.politics}</span></div>
                            <div class="char-stat"><span>CHA</span><span>${s.charisma}</span></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    DOM.characterList.innerHTML = html || '<div style="color:#8890a8;font-size:13px">No officers</div>';
}

function updateDiplomacy() {
    if (!G.state || !G.factionId) { DOM.diplomacyList.innerHTML = ''; return; }

    const diplomacy = G.state.diplomacy;
    let html = '';

    Object.values(diplomacy).forEach(d => {
        let otherId = null;
        if (d.faction_a_id === G.factionId) otherId = d.faction_b_id;
        else if (d.faction_b_id === G.factionId) otherId = d.faction_a_id;
        else return;

        const other = G.state.factions[otherId];
        if (!other) return;

        html += `
            <div class="diplo-item">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:${other.color};font-weight:bold">${other.name}</span>
                    <span class="diplo-status ${d.status}">${d.status}</span>
                </div>
                <div style="font-size:12px;color:#a0a0b0;margin-top:4px">
                    Opinion: ${d.opinion} | Trade: ${Math.floor(d.trade_value)}
                    ${d.truce_turns > 0 ? ` | Truce: ${d.truce_turns} turns` : ''}
                </div>
                <div class="diplo-actions">
                    <button class="btn btn-small" onclick="diplomacyAction('${otherId}','ally')">Ally</button>
                    <button class="btn btn-small" onclick="diplomacyAction('${otherId}','trade')">Trade</button>
                    <button class="btn btn-small btn-danger" onclick="diplomacyAction('${otherId}','war')">War</button>
                    <button class="btn btn-small" onclick="diplomacyAction('${otherId}','truce')">Truce</button>
                </div>
            </div>
        `;
    });
    DOM.diplomacyList.innerHTML = html || '<div style="color:#a0a0b0;font-size:13px">No relations</div>';
}

function updateTechPanel() {
    if (!G.state || !G.factionId) return;
    const faction = G.state.factions[G.factionId];
    if (!faction) return;

    let infoHtml = `
        <div class="detail-row"><span class="detail-label">Military</span><span class="detail-value">Lv ${faction.tech_levels.military}</span></div>
        <div class="detail-row"><span class="detail-label">Economy</span><span class="detail-value">Lv ${faction.tech_levels.economy}</span></div>
        <div class="detail-row"><span class="detail-label">Culture</span><span class="detail-value">Lv ${faction.tech_levels.culture}</span></div>
        <div class="detail-row"><span class="detail-label">Diplomacy</span><span class="detail-value">Lv ${faction.tech_levels.diplomacy}</span></div>
        <div class="detail-row"><span class="detail-label">Research Points</span><span class="detail-value">${Math.floor(faction.research_points)}</span></div>
    `;
    if (faction.current_research) {
        infoHtml += `<div class="detail-row"><span class="detail-label">Researching</span><span class="detail-value" style="color:#f5c518">${faction.current_research}</span></div>`;
    }
    DOM.techInfo.innerHTML = infoHtml;

    // Fetch available techs
    fetch(`/api/rooms/${G.roomId}/available_techs/${G.factionId}`)
        .then(r => r.json())
        .then(data => {
            if (!data.techs) return;
            let html = '<div style="margin-top:8px;font-size:12px;color:#a0a0b0">Available Research:</div>';
            data.techs.forEach(t => {
                const isResearching = faction.current_research === t.id;
                html += `
                    <div class="tech-item ${isResearching ? 'researching' : ''}" onclick="researchTech('${t.id}')">
                        <div style="display:flex;justify-content:space-between">
                            <span class="tech-name">${t.name}</span>
                            <span class="tech-cost">Cost: ${t.cost}</span>
                        </div>
                        <div class="tech-desc">${t.description}</div>
                        <div style="font-size:11px;color:#9b59b6;margin-top:2px">[${t.category}] Lv ${t.level}</div>
                    </div>
                `;
            });
            DOM.techAvailable.innerHTML = html;
        })
        .catch(() => {});
}

// ============================================================
// Map Rendering
// ============================================================

let ctx = null;

function resizeCanvas() {
    const container = $('#map-container');
    DOM.canvas.width = container.clientWidth;
    DOM.canvas.height = container.clientHeight;
    ctx = DOM.canvas.getContext('2d');
}

function hexToPixel(col, row) {
    const x = HEX.size * 1.5 * col;
    const y = HEX.height * (row + 0.5 * (col % 2));
    return { x, y };
}

function pixelToHex(px, py) {
    // Approximate inverse
    const col = Math.round(px / (HEX.size * 1.5));
    const row = Math.round((py / HEX.height) - 0.5 * (col % 2));
    return { col, row };
}

function drawHex(cx, cy, size, fillColor, strokeColor, lineWidth) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 180) * (60 * i - 30);
        const hx = cx + size * Math.cos(angle);
        const hy = cy + size * Math.sin(angle);
        if (i === 0) ctx.moveTo(hx, hy);
        else ctx.lineTo(hx, hy);
    }
    ctx.closePath();

    if (fillColor) {
        ctx.fillStyle = fillColor;
        ctx.fill();
    }
    if (strokeColor) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth || 1;
        ctx.stroke();
    }
}

function renderMap() {
    if (!ctx || !G.state) return;
    const W = DOM.canvas.width;
    const H = DOM.canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(G.mapOffset.x + W / 2, G.mapOffset.y + H / 2);
    ctx.scale(G.mapZoom, G.mapZoom);

    // Center the map
    const centerX = (G.state._meta_map_cols || 10) * HEX.size * 1.5 / 2;
    const centerY = (G.state._meta_map_rows || 8) * HEX.height / 2;
    ctx.translate(-centerX, -centerY);

    const provinces = G.state.provinces;

    // Draw provinces
    Object.values(provinces).forEach(p => {
        const pos = hexToPixel(p.x, p.y);

        // Base terrain color
        let fill = TERRAIN_COLORS[p.terrain] || '#555';

        // Tint with faction color if owned
        if (p.owner_faction_id && G.state.factions[p.owner_faction_id]) {
            const factionColor = G.state.factions[p.owner_faction_id].color;
            fill = blendColors(fill, factionColor, 0.35);
        }

        let strokeCol = '#1a1a2e';
        let lineW = 1;

        if (G.selectedProvince === p.id) {
            strokeCol = '#f5c518';
            lineW = 3;
        } else if (G.hoveredProvince === p.id) {
            strokeCol = '#ffffff';
            lineW = 2;
        }

        drawHex(pos.x, pos.y, HEX.size - 1, fill, strokeCol, lineW);

        // Province name
        ctx.fillStyle = '#ffffff';
        ctx.font = `${Math.max(8, 10 * G.mapZoom)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(p.name.substring(0, 8), pos.x, pos.y - 8);

        // Terrain icon / info
        ctx.font = '9px sans-serif';
        ctx.fillStyle = '#cccccc';
        ctx.fillText(`Pop:${Math.floor(p.population / 1000)}k`, pos.x, pos.y + 6);

        // Fortification indicator
        if (p.fortification > 0) {
            ctx.fillStyle = '#f39c12';
            ctx.fillText(`Fort:${p.fortification}`, pos.x, pos.y + 16);
        }

        // Army indicators
        const armiesHere = Object.values(G.state.armies).filter(a => a.province_id === p.id);
        if (armiesHere.length > 0) {
            armiesHere.forEach((a, idx) => {
                const faction = G.state.factions[a.faction_id];
                const color = faction ? faction.color : '#fff';
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(pos.x - 12 + idx * 10, pos.y + 22, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#fff';
                ctx.font = '7px sans-serif';
                ctx.fillText(Math.floor(a.soldiers / 100), pos.x - 12 + idx * 10, pos.y + 30);
            });
        }
    });

    ctx.restore();
}

function blendColors(hex1, hex2, ratio) {
    const c1 = hexToRgb(hex1);
    const c2 = hexToRgb(hex2);
    if (!c1 || !c2) return hex1;
    const r = Math.round(c1.r * (1 - ratio) + c2.r * ratio);
    const g = Math.round(c1.g * (1 - ratio) + c2.g * ratio);
    const b = Math.round(c1.b * (1 - ratio) + c2.b * ratio);
    return `rgb(${r},${g},${b})`;
}

function hexToRgb(hex) {
    if (hex.startsWith('rgb')) {
        const m = hex.match(/(\d+)/g);
        if (m) return { r: +m[0], g: +m[1], b: +m[2] };
        return null;
    }
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
    } : null;
}

// ============================================================
// Map Interaction
// ============================================================

function getProvinceAtPixel(canvasX, canvasY) {
    if (!G.state) return null;
    const W = DOM.canvas.width;
    const H = DOM.canvas.height;

    // Reverse the canvas transforms
    let x = canvasX - G.mapOffset.x - W / 2;
    let y = canvasY - G.mapOffset.y - H / 2;
    x /= G.mapZoom;
    y /= G.mapZoom;

    const centerX = (G.state._meta_map_cols || 10) * HEX.size * 1.5 / 2;
    const centerY = (G.state._meta_map_rows || 8) * HEX.height / 2;
    x += centerX;
    y += centerY;

    // Find closest province
    let closest = null;
    let minDist = Infinity;

    Object.values(G.state.provinces).forEach(p => {
        const pos = hexToPixel(p.x, p.y);
        const dx = pos.x - x;
        const dy = pos.y - y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < HEX.size && dist < minDist) {
            minDist = dist;
            closest = p.id;
        }
    });

    return closest;
}

DOM.canvas.addEventListener('mousedown', (e) => {
    G.isDragging = true;
    G.dragStart = { x: e.clientX - G.mapOffset.x, y: e.clientY - G.mapOffset.y };
});

DOM.canvas.addEventListener('mousemove', (e) => {
    if (G.isDragging) {
        G.mapOffset.x = e.clientX - G.dragStart.x;
        G.mapOffset.y = e.clientY - G.dragStart.y;
        renderMap();
    } else {
        const rect = DOM.canvas.getBoundingClientRect();
        const pid = getProvinceAtPixel(e.clientX - rect.left, e.clientY - rect.top);
        if (pid !== G.hoveredProvince) {
            G.hoveredProvince = pid;
            renderMap();
        }
    }
});

DOM.canvas.addEventListener('mouseup', (e) => {
    if (G.isDragging) {
        const dx = e.clientX - G.dragStart.x - G.mapOffset.x;
        const dy = e.clientY - G.dragStart.y - G.mapOffset.y;
        // If barely moved, treat as click
        if (Math.abs(dx) < 5 && Math.abs(dy) < 5) {
            // Actually this logic is slightly off; let's use a separate click
        }
    }
    G.isDragging = false;
});

DOM.canvas.addEventListener('click', (e) => {
    const rect = DOM.canvas.getBoundingClientRect();
    const pid = getProvinceAtPixel(e.clientX - rect.left, e.clientY - rect.top);
    if (pid) {
        G.selectedProvince = pid;
        updateSidePanel();
        updateArmyList();
        renderMap();
    }
});

DOM.canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    G.mapZoom = Math.max(0.3, Math.min(3.0, G.mapZoom + delta));
    renderMap();
});

window.addEventListener('resize', () => {
    if (!DOM.gameScreen.classList.contains('hidden')) {
        resizeCanvas();
        renderMap();
    }
});

// ============================================================
// Player Actions
// ============================================================

function buildInProvince(buildingType) {
    if (!G.selectedProvince) return;
    sendWS({
        type: 'action',
        action: {
            type: 'build',
            province_id: G.selectedProvince,
            building_type: buildingType,
        }
    });
    // Optimistic: request state update
    setTimeout(() => sendWS({ type: 'get_state' }), 300);
}

function recruitArmy() {
    if (!G.selectedProvince) return;
    sendWS({
        type: 'action',
        action: {
            type: 'recruit_army',
            province_id: G.selectedProvince,
            soldiers: 1000,
        }
    });
    setTimeout(() => sendWS({ type: 'get_state' }), 300);
}

function recruitCharacter() {
    if (!G.selectedProvince) return;
    sendWS({
        type: 'action',
        action: {
            type: 'recruit_character',
            province_id: G.selectedProvince,
        }
    });
    setTimeout(() => sendWS({ type: 'get_state' }), 300);
}

function moveArmy(armyId, targetProvinceId) {
    sendWS({
        type: 'action',
        action: {
            type: 'move_army',
            army_id: armyId,
            target_province_id: targetProvinceId,
        }
    });
    setTimeout(() => sendWS({ type: 'get_state' }), 300);
}

function selectArmy(armyId) {
    G.selectedArmy = armyId;
    const army = G.state.armies[armyId];
    if (army) {
        G.selectedProvince = army.province_id;
        updateSidePanel();
        updateArmyList();
        renderMap();
    }
}

function diplomacyAction(targetFactionId, action) {
    sendWS({
        type: 'action',
        action: {
            type: 'diplomacy',
            target_faction_id: targetFactionId,
            action: action,
        }
    });
    setTimeout(() => sendWS({ type: 'get_state' }), 300);
}

function researchTech(techId) {
    sendWS({
        type: 'action',
        action: {
            type: 'research',
            tech_id: techId,
        }
    });
    setTimeout(() => sendWS({ type: 'get_state' }), 300);
}

function endTurn() {
    sendWS({ type: 'ready' });
    // Also force turn for single-player / testing
    sendWS({ type: 'force_turn' });
}

function saveGame() {
    fetch(`/api/rooms/${G.roomId}/save`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.ok) addChatMessage('System', 'Game saved!', '#2ecc71');
            else addChatMessage('System', 'Save failed', '#e74c3c');
        });
}

// ============================================================
// Events & Chat
// ============================================================

function addEvents(events) {
    events.forEach(evt => {
        const div = document.createElement('div');
        div.className = `event-entry event-type-${evt.event_type}`;
        div.innerHTML = `<span class="event-turn">[${evt.turn}]</span> ${evt.description}`;
        DOM.eventList.appendChild(div);
    });
    DOM.eventList.scrollTop = DOM.eventList.scrollHeight;
}

function addChatMessage(name, text, color, factionId) {
    const div = document.createElement('div');
    div.className = 'chat-msg';
    const nameColor = color || (factionId && G.state && G.state.factions[factionId]
        ? G.state.factions[factionId].color : '#f5c518');
    div.innerHTML = `<span class="chat-name" style="color:${nameColor}">${name}:</span> <span class="chat-text">${escapeHtml(text)}</span>`;
    DOM.chatMessages.appendChild(div);
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
}

function sendChat() {
    const text = DOM.chatInput.value.trim();
    if (!text) return;
    sendWS({ type: 'chat', text });
    DOM.chatInput.value = '';
}

// ============================================================
// Tab Management
// ============================================================

$$('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        $$('.tab').forEach(t => t.classList.remove('active'));
        $$('.tab-content').forEach(tc => tc.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.dataset.tab;
        const content = $(`#${target}`);
        if (content) content.classList.add('active');
    });
});

// ============================================================
// Utilities
// ============================================================

function capitalize(s) {
    if (!s) return '';
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// Event Listeners
// ============================================================

DOM.btnConnect.addEventListener('click', connect);
DOM.btnStartGame.addEventListener('click', startGame);
DOM.btnEndTurn.addEventListener('click', endTurn);
DOM.btnSave.addEventListener('click', saveGame);
DOM.btnChatSend.addEventListener('click', sendChat);

DOM.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChat();
});

DOM.playerName.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') connect();
});

// Auto-detect server URL
if (!DOM.serverUrl.value) {
    const loc = window.location;
    DOM.serverUrl.value = `${loc.protocol === 'https:' ? 'wss:' : 'ws:'}//${loc.host}`;
}

// ============================================================
// Minimap
// ============================================================

let minimapCtx = null;

function initMinimap() {
    const mc = document.getElementById('minimap');
    if (!mc) return;
    const container = document.getElementById('minimap-container');
    mc.width = container.clientWidth;
    mc.height = container.clientHeight;
    minimapCtx = mc.getContext('2d');
}

function renderMinimap() {
    if (!minimapCtx || !G.state) return;
    const mc = document.getElementById('minimap');
    const W = mc.width;
    const H = mc.height;

    minimapCtx.clearRect(0, 0, W, H);
    minimapCtx.fillStyle = 'rgba(10,10,26,0.9)';
    minimapCtx.fillRect(0, 0, W, H);

    const provinces = G.state.provinces;
    if (!provinces) return;

    // Find map bounds
    let maxX = 0, maxY = 0;
    Object.values(provinces).forEach(p => {
        maxX = Math.max(maxX, p.x);
        maxY = Math.max(maxY, p.y);
    });

    const padX = 12, padY = 12;
    const scaleX = (W - padX * 2) / ((maxX + 1) * 1.5);
    const scaleY = (H - padY * 2) / ((maxY + 1) * Math.sqrt(3) * 0.5 + 1);
    const scale = Math.min(scaleX, scaleY);

    Object.values(provinces).forEach(p => {
        const px = padX + p.x * 1.5 * scale;
        const py = padY + (p.y + 0.5 * (p.x % 2)) * Math.sqrt(3) * 0.5 * scale;

        let color = TERRAIN_COLORS[p.terrain] || '#555';
        if (p.owner_faction_id && G.state.factions[p.owner_faction_id]) {
            color = blendColors(color, G.state.factions[p.owner_faction_id].color, 0.5);
        }

        minimapCtx.fillStyle = color;
        minimapCtx.beginPath();
        minimapCtx.arc(px, py, Math.max(2, scale * 0.4), 0, Math.PI * 2);
        minimapCtx.fill();

        // Highlight selected
        if (G.selectedProvince === p.id) {
            minimapCtx.strokeStyle = '#f5c518';
            minimapCtx.lineWidth = 2;
            minimapCtx.beginPath();
            minimapCtx.arc(px, py, Math.max(4, scale * 0.6), 0, Math.PI * 2);
            minimapCtx.stroke();
        }
    });

    // Draw armies as small dots
    if (G.state.armies) {
        Object.values(G.state.armies).forEach(a => {
            const p = provinces[a.province_id];
            if (!p) return;
            const px = padX + p.x * 1.5 * scale;
            const py = padY + (p.y + 0.5 * (p.x % 2)) * Math.sqrt(3) * 0.5 * scale;
            const faction = G.state.factions[a.faction_id];
            minimapCtx.fillStyle = faction ? faction.color : '#fff';
            minimapCtx.fillRect(px - 1, py + 3, 3, 3);
        });
    }
}

// ============================================================
// Notifications
// ============================================================

function showNotification(text, type = 'info') {
    const area = document.getElementById('notification-area');
    if (!area) return;

    const div = document.createElement('div');
    div.className = `notification ${type}`;

    const icons = { info: 'i', success: '!', warning: '?', error: 'X' };
    div.innerHTML = `<span class="notification-icon">${icons[type] || 'i'}</span><span>${escapeHtml(text)}</span>`;

    area.appendChild(div);

    // Auto remove after 4 seconds
    setTimeout(() => {
        div.classList.add('fade-out');
        setTimeout(() => div.remove(), 300);
    }, 4000);
}

// ============================================================
// Enhanced Map Rendering
// ============================================================

function renderMapEnhanced() {
    renderMap();
    renderMinimap();
}

// Override updateAll to use enhanced rendering
const _origUpdateAll = updateAll;
updateAll = function() {
    _origUpdateAll();
    renderMinimap();
};

// ============================================================
// Initialize
// ============================================================

// Setup minimap when game starts
const _origShowScreen = showScreen;
showScreen = function(name) {
    _origShowScreen(name);
    if (name === 'game') {
        setTimeout(() => {
            initMinimap();
            renderMinimap();
        }, 100);
    }
};

console.log('Civilization Online RPG Client v0.2.0 loaded');
