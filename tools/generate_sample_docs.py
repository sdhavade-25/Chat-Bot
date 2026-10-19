"""Generate the fictional sample documents in sample_docs/.

The PDFs imitate a PLM/CAD user-manual layout (numbered headings, UI screenshots)
so the whole RAG pipeline can be demonstrated without any real company data.
Everything here is invented: the company, the people and the procedures.

Run:  py tools/generate_sample_docs.py        (needs Pillow, in requirements-dev.txt)
"""
import shutil
import textwrap
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_docs"
TMP_SHOTS = OUT_DIR / "_shots"

PAGE_W, PAGE_H = 595, 842          # A4 in points
MARGIN = 56
ACCENT = (31, 111, 235)


def rgb(color):
    """PyMuPDF wants colour components in 0..1."""
    return tuple(c / 255 for c in color)


# ---------------------------------------------------------------- fake UI screenshots

def make_screenshot(path, title, rows, accent=ACCENT):
    """Draw a fake application window so the PDFs contain real image blocks."""
    img = Image.new("RGB", (800, 500), (240, 242, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 800, 40], fill=(32, 36, 42))
    d.text((14, 13), f"Fabrikam PLM  -  {title}", fill=(255, 255, 255))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([712 + i * 26, 13, 726 + i * 26, 27], fill=c)
    d.rectangle([0, 40, 220, 500], fill=(251, 252, 254), outline=(214, 219, 228))
    y = 60
    for item in ["Home", "My Tasks", "Items", "Datasheets", "Reports"]:
        d.rectangle([14, y, 206, y + 30],
                    fill=(232, 240, 254) if item == "My Tasks" else (251, 252, 254),
                    outline=(220, 224, 232))
        d.text((26, y + 8), item, fill=(40, 44, 52))
        y += 38
    d.rectangle([240, 60, 780, 100], fill=accent)
    d.text((254, 73), title, fill=(255, 255, 255))
    y = 120
    for row in rows:
        d.rectangle([240, y, 780, y + 34], fill=(255, 255, 255), outline=(216, 220, 228))
        d.text((254, y + 9), row, fill=(45, 48, 56))
        y += 42
    img.save(path)


# ---------------------------------------------------------------- PDF layout helpers

def heading(page, y, text):
    page.insert_text((MARGIN, y), text, fontsize=15, fontname="hebo", color=rgb(ACCENT))
    page.draw_line(pymupdf.Point(MARGIN, y + 8), pymupdf.Point(PAGE_W - MARGIN, y + 8),
                   color=rgb((200, 206, 216)), width=0.8)
    return y + 32


def para(page, y, text, width=92):
    for line in textwrap.wrap(text, width=width):
        page.insert_text((MARGIN, y), line, fontsize=10.5, fontname="helv", color=rgb((35, 38, 45)))
        y += 15
    return y + 8


def bullets(page, y, items, width=88):
    for item in items:
        lines = textwrap.wrap(item, width=width)
        for n, line in enumerate(lines):
            page.insert_text((MARGIN + 12, y), ("- " if n == 0 else "  ") + line,
                             fontsize=10.5, fontname="helv", color=rgb((35, 38, 45)))
            y += 15
        y += 4
    return y + 4


def figure(page, y, shot_path):
    rect = pymupdf.Rect(MARGIN, y, MARGIN + 300, y + 187.5)
    page.insert_image(rect, filename=str(shot_path))
    return y + 187.5 + 24


def cover(doc, title, subtitle):
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_rect(pymupdf.Rect(0, 0, PAGE_W, 120), color=None, fill=rgb((32, 36, 42)))
    page.insert_text((MARGIN, 70), "FABRIKAM INDUSTRIES", fontsize=20, fontname="hebo",
                     color=rgb((255, 255, 255)))
    page.insert_text((MARGIN, 200), title, fontsize=26, fontname="hebo", color=rgb(ACCENT))
    page.insert_text((MARGIN, 232), subtitle, fontsize=13, fontname="helv", color=rgb((80, 86, 96)))
    page.insert_text((MARGIN, 290), "Revision 2.1", fontsize=11, fontname="helv", color=rgb((80, 86, 96)))
    page.insert_text((MARGIN, 308), "PLM Support: plm-support@fabrikam.example  (555-0142)",
                     fontsize=11, fontname="helv", color=rgb((80, 86, 96)))
    page.insert_text((MARGIN, 760), "FICTIONAL sample document, generated for demo purposes.",
                     fontsize=9, fontname="heit", color=rgb((140, 145, 155)))
    return doc


# ---------------------------------------------------------------- the documents

def teamcenter_guide(shots):
    doc = cover(pymupdf.open(), "Teamcenter User Guide", "Everyday PLM tasks for design engineers")
    sections = [
        ("01 - LOGGING IN TO TEAMCENTER", [
            "Open your browser and go to teamcenter.fabrikam.example. Sign in with your network "
            "account. Your initial password is set by the PLM administrator and must be changed at "
            "first login; passwords expire every 90 days. If you forget your password, use the "
            "'Request a reset' link on the login page or contact the PLM administrator.",
        ], shots["login"]),
        ("02 - CREATING A NEW ITEM", [
            "From Home choose Create > Item. Give the item a name and select an item type: use Part "
            "for a single component, Product for an assembly, and Document for drawings or specs. "
            "The PLM administrator can assign additional item types for your organisation.",
        ], None),
        ("03 - SAVE AS", [
            "Use File > Save As to save an existing item as a new item. Save As creates a new item "
            "ID and copies every attached dataset to it; the original item is not modified. Give the "
            "new item a meaningful name before you confirm, because the item ID cannot be changed "
            "afterwards.",
        ], shots["saveas"]),
        ("04 - CREATE NEW REVISION", [
            "Select the item and click Revise in the toolbar. Teamcenter creates the next revision "
            "letter (for example A becomes B), copies all datasets forward and makes the new "
            "revision the only editable one. The previous revision stays visible for reference but "
            "is locked.",
        ], None),
        ("05 - RELEASE WORKFLOWS", [
            "A release workflow can only be initiated by a user with the Creator role. Select the "
            "item, choose Add to Workflow > Release Workflow, add the required sign-off roles and "
            "click Start. Every approver is notified in their My Tasks list.",
        ], shots["workflow"]),
        ("06 - REJECTING A WORKFLOW TASK", [
            "When the Checker rejects a task, the workflow sends the item back to the Creator for "
            "self review. The task reappears in the Creator's My Tasks list with the Checker's "
            "comments, and the workflow restarts at the review step once the Creator submits it "
            "again.",
        ], None),
        ("07 - COMMENTS ON WORKFLOW TASKS", [
            "A comment is mandatory when rejecting a workflow task, so the next person always knows "
            "what has to be fixed. When approving a task a comment is optional. Comments are stored "
            "in the item's workflow history and can be read later from View Revisions > Workflow "
            "History.",
        ], None),
        ("08 - EXPORTING A BOM TO EXCEL", [
            "Open the assembly and switch to the Structures tab. Select the root node and use the "
            "Excel Round Trip command to export the BOM to Excel. Edit quantities or part numbers "
            "in the spreadsheet, save it, then use Excel Round Trip > Import to write the changes "
            "back into Teamcenter. Row formatting from your template is preserved.",
        ], shots["bom"]),
        ("09 - COMPARING ASSEMBLY STRUCTURES", [
            "Open two structures and use Split Context to display them side by side. The comparison "
            "highlights differences in quantity, revision and position, so you can see exactly what "
            "changed between two releases of the same assembly.",
        ], shots["split"]),
        ("10 - REVISION RULES", [
            "Set the revision rule in the toolbar before you open an assembly. The Latest Working "
            "revision rule shows the latest revision of the child parts, even if they are still "
            "being edited. Use Release Baseline when you need the exact configuration that was "
            "released.",
        ], None),
    ]
    for title, paras, shot in sections:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = heading(page, 80, title)
        for text in paras:
            y = para(page, y, text)
        if shot:
            y = figure(page, y + 8, shot)
    return doc


def creo_guide(shots):
    doc = cover(pymupdf.open(), "Creo Parametric User Guide", "CAD setup and data safety for Creo users")
    sections = [
        ("01 - SELECTING A ROLE", [
            "When Creo starts, choose a role from the role selector. Select the Creator role to "
            "create models and drawings. The Checker and Approver roles are review-only and hide "
            "the modelling commands. Your role also decides which templates and layers are "
            "available in new files.",
        ], shots["roles"]),
        ("02 - CHECKING OUT DEPENDENT FILES", [
            "Creo asks you to check out the model before saving a drawing because the drawing "
            "references model geometry as dependent files. Checking out locks the model for you, "
            "lets the drawing store the new references, and unlocks the model again when you check "
            "it back in. Never skip this step on shared assemblies.",
        ], None),
    ]
    for title, paras, shot in sections:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = heading(page, 80, title)
        for text in paras:
            y = para(page, y, text)
        if shot:
            y = figure(page, y + 8, shot)
    return doc


def solidworks_guide(shots):
    doc = cover(pymupdf.open(), "SolidWorks Best Practices", "Saving CAD data through the PLM connector")
    sections = [
        ("01 - ITEM TYPES IN SOLIDWORKS", [
            "When you create a part in SolidWorks and save it to Teamcenter through the connector, "
            "the item type Semi-Finished Component is assigned automatically. Assemblies are stored "
            "with the item type Product, and drawings with Drawing. Do not change these types "
            "manually; downstream reports rely on them.",
        ], None),
        ("02 - SAVING AN ASSEMBLY", [
            "Save the top-level assembly first, then resolve any missing references in the order "
            "the connector lists them. Saving child parts first confuses the structure check and "
            "can create duplicate items.",
        ], shots["sw_save"]),
    ]
    for title, paras, shot in sections:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = heading(page, 80, title)
        for text in paras:
            y = para(page, y, text)
        if shot:
            y = figure(page, y + 8, shot)
    return doc


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    TMP_SHOTS.mkdir(parents=True)
    shots = {
        "login": TMP_SHOTS / "login.png",
        "saveas": TMP_SHOTS / "saveas.png",
        "workflow": TMP_SHOTS / "workflow.png",
        "bom": TMP_SHOTS / "bom.png",
        "split": TMP_SHOTS / "split.png",
        "roles": TMP_SHOTS / "roles.png",
        "sw_save": TMP_SHOTS / "sw_save.png",
    }
    make_screenshot(shots["login"], "Sign in", ["User name:  d.former", "Password:  ********",
                                                "Domain:  FABRIKAM", "[ Sign in ]  [ Request a reset ]"])
    make_screenshot(shots["saveas"], "Save As - New Item", ["Name:  Bracket, Mounting (Left)",
                                                            "Item type:  Part",
                                                            "Item ID:  FAB-000482 (auto)",
                                                            "[ OK ]  [ Cancel ]"])
    make_screenshot(shots["workflow"], "Release Workflow", ["Step 1  Creator      - submitted",
                                                            "Step 2  Self Review  - in progress",
                                                            "Step 3  Checker     - pending",
                                                            "Step 4  Approver    - pending"])
    make_screenshot(shots["bom"], "Excel Round Trip", ["Export BOM to Excel (.xlsx)",
                                                       "Template:  fabrikam_bom_v3.xlsx",
                                                       "Rows:  148",
                                                       "[ Export ]  [ Import ]"])
    make_screenshot(shots["split"], "Split Context", ["Assembly  FAB-1001  rev C  |  rev D",
                                                      "Qty 4  ->  Qty 6          differs",
                                                      "Bracket  rev B  ->  rev C  differs",
                                                      "Pin, Dowel  rev A          same"])
    make_screenshot(shots["roles"], "Creo Role Selector", ["Role:  Creator      (models and drawings)",
                                                           "Role:  Checker      (review only)",
                                                           "Role:  Approver     (review only)",
                                                           "[ OK ]"])
    make_screenshot(shots["sw_save"], "SolidWorks - Save to Teamcenter",
                    ["Resolving references for  FAB-2200.SLDASM",
                     "1. FAB-2201.SLDPRT   ok",
                     "2. FAB-2202.SLDPRT   missing - browse",
                     "[ Save all ]  [ Cancel ]"])

    outputs = {
        "Fabrikam_Teamcenter_User_Guide.pdf": teamcenter_guide(shots),
        "Fabrikam_Creo_User_Guide.pdf": creo_guide(shots),
        "Fabrikam_SolidWorks_Best_Practices.pdf": solidworks_guide(shots),
    }
    for name, doc in outputs.items():
        doc.save(OUT_DIR / name, garbage=3, deflate=True)
        doc.close()
        print(f"wrote {OUT_DIR / name}")
    shutil.rmtree(TMP_SHOTS)


if __name__ == "__main__":
    main()
