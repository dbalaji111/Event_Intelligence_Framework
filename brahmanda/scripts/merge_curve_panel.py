"""
Merges curve_panel.html into redflag_dashboard_v5.html, inserting it
right before the final </body> tag. Run once from the repo root.

    python scripts/merge_curve_panel.py

Idempotent: if curveSection is already present in the dashboard file,
it skips the merge instead of duplicating the panel.
"""
from pathlib import Path

DASHBOARD_PATH = Path("redflag_dashboard_v5.html")   # adjust path if it lives elsewhere
CURVE_PANEL_PATH = Path("scripts/curve_panel.html")   # wherever you saved the panel file

def main():
    if not DASHBOARD_PATH.exists():
        print(f"ERROR: {DASHBOARD_PATH} not found. Run this from the repo root, "
              f"or edit DASHBOARD_PATH at the top of this script.")
        return
    if not CURVE_PANEL_PATH.exists():
        print(f"ERROR: {CURVE_PANEL_PATH} not found. Save curve_panel.html there first.")
        return

    dashboard = DASHBOARD_PATH.read_text(encoding="utf-8")
    panel = CURVE_PANEL_PATH.read_text(encoding="utf-8")

    if 'id="curveSection"' in dashboard:
        print("curveSection already present in the dashboard -- skipping merge "
              "(delete it manually first if you want to re-merge a newer panel).")
        return

    if "</body>" not in dashboard:
        print("ERROR: no </body> tag found in the dashboard file -- can't merge safely.")
        return

    merged = dashboard.replace("</body>", panel + "\n</body>")

    backup_path = DASHBOARD_PATH.with_suffix(".html.bak")
    backup_path.write_text(dashboard, encoding="utf-8")
    DASHBOARD_PATH.write_text(merged, encoding="utf-8")

    print(f"Backup saved to {backup_path}")
    print(f"Merged {CURVE_PANEL_PATH.name} into {DASHBOARD_PATH.name}")
    print(f"Dashboard size: {len(dashboard)} -> {len(merged)} chars")

if __name__ == "__main__":
    main()
