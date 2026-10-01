"""Team colors + badge chips for the Streamlit app.

TEAM_COLORS maps nflverse abbreviation -> (primary, secondary) hex.
Badge = tight inline-flex span (display only, not clickable): padding 0,
margin 0, line-height 1, monospace bold, ~4px radius, subtle box-shadow,
shrink-wrapped to the text. NAV buttons use st-button with per-team CSS
scoped via Streamlit's st-key-<key> classes.
"""

TEAM_COLORS = {
    'ARI': ('#97233F', '#FFFFFF'), 'ATL': ('#A71930', '#FFFFFF'),
    'BAL': ('#241773', '#FFFFFF'), 'BUF': ('#00338D', '#FFFFFF'),
    'CAR': ('#0085CA', '#FFFFFF'), 'CHI': ('#0B162A', '#FFFFFF'),
    'CIN': ('#FB4F14', '#000000'), 'CLE': ('#311D00', '#FFFFFF'),
    'DAL': ('#003594', '#FFFFFF'), 'DEN': ('#FB4F14', '#002244'),
    'DET': ('#0076B6', '#FFFFFF'), 'GB':  ('#203731', '#FFB612'),
    'HOU': ('#03202F', '#A71930'), 'IND': ('#002C5F', '#FFFFFF'),
    'JAX': ('#006778', '#D7A22A'), 'KC':  ('#E31837', '#FFB612'),
    'LV':  ('#000000', '#A5ACAF'), 'LAC': ('#0080C6', '#FFC20E'),
    'LAR': ('#003594', '#FFA300'), 'MIA': ('#008E97', '#FC4C02'),
    'MIN': ('#4F2683', '#FFC62F'), 'NE':  ('#002244', '#C60C30'),
    'NO':  ('#D3BC8D', '#000000'), 'NYG': ('#0B2265', '#A71930'),
    'NYJ': ('#125740', '#FFFFFF'), 'PHI': ('#004C54', '#A5ACAF'),
    'PIT': ('#FFB612', '#000000'), 'SF':  ('#AA0000', '#B3995D'),
    'SEA': ('#002244', '#69BE28'), 'TB':  ('#D50A0A', '#FF7900'),
    'TEN': ('#0C2340', '#4B92DB'), 'WAS': ('#5A1414', '#FFB612'),
}
# nflverse uses LA for the Rams; the color table lists LAR
TEAM_COLORS["LA"] = TEAM_COLORS["LAR"]

TEAM_ORDER = sorted(a for a in TEAM_COLORS if a != "LAR")

BADGE_CSS = """<style>
.team-badge{display:inline-flex;align-items:center;justify-content:center;
padding:0;margin:0;line-height:1;font-family:ui-monospace,Menlo,monospace;
font-weight:700;font-size:13px;border-radius:4px;
box-shadow:0 1px 2px rgba(0,0,0,.35);width:max-content;}
</style>"""


def badge_html(abbr):
    bg, fg = TEAM_COLORS.get(abbr, ("#333333", "#FFFFFF"))
    return f'<span class="team-badge" style="background:{bg};color:{fg}">{abbr}</span>'


def nav_button_css():
    """Per-team colors for the nav grid's st.buttons, scoped via the
    st-key-btn_<abbr> classes Streamlit adds for keyed widgets."""
    rules = "\n".join(
        f".st-key-btn_{a} button{{background:{TEAM_COLORS[a][0]}!important;"
        f"color:{TEAM_COLORS[a][1]}!important;border:none!important;}}"
        for a in TEAM_ORDER
    )
    return f"<style>{rules}</style>"
