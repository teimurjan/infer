---
name: infer
description: A pocket dictionary of AI models that run in Safari on the GPU of an iPhone.
colors:
  paper: "#fafaf7"
  ink: "#0b0b0c"
  ink-2: "#3a3a37"
  ink-3: "#6b6b66"
  hair: "#d8d8d1"
  live: "#1d2bd9"
  live-wash: "#1d2bd914"
  fail: "#c4161c"
typography:
  display:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(6rem, 33vw, 15rem)"
    fontWeight: 900
    lineHeight: 0.8
    letterSpacing: "-0.06em"
  headline:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(2.75rem, 12vw, 5.5rem)"
    fontWeight: 900
    lineHeight: 0.88
    letterSpacing: "-0.045em"
  title:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 850
    lineHeight: 0.92
    letterSpacing: "-0.035em"
  term:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 850
    lineHeight: 1
    letterSpacing: "-0.01em"
  sense:
    fontFamily: "Source Serif 4 Variable, Georgia, Times New Roman, serif"
    fontSize: "1.1875rem"
    fontWeight: 400
    lineHeight: 1.42
  body:
    fontFamily: "Source Serif 4 Variable, Georgia, Times New Roman, serif"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.45
    fontFeature: "\"onum\" 1, \"pnum\" 1"
  label:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 750
    lineHeight: 1
    letterSpacing: "0.02em"
    fontVariation: "\"wdth\" 80"
  control:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.01em"
  number:
    fontFamily: "Archivo Variable, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 500
    lineHeight: 1.6
    fontFeature: "\"lnum\" 1, \"tnum\" 1"
    fontVariation: "\"wdth\" 92"
  citation:
    fontFamily: "Source Serif 4 Variable, Georgia, Times New Roman, serif"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.45
  data:
    fontFamily: "ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.55
rounded:
  none: "0px"
  tab: "5px"
spacing:
  s-1: "4px"
  s-2: "8px"
  s-3: "12px"
  s-4: "16px"
  s-5: "24px"
  s-6: "32px"
  s-7: "48px"
  s-8: "72px"
  gutter: "18px"
  gutter-wide: "40px"
  fore-edge: "46px"
  fore-edge-wide: "72px"
components:
  button-primary:
    backgroundColor: "{colors.live}"
    textColor: "{colors.paper}"
    typography: "{typography.control}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "44px"
  button-primary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  button-ink:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.control}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "44px"
  button-ink-hover:
    backgroundColor: "{colors.ink-2}"
    textColor: "{colors.paper}"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.none}"
    padding: "0 16px"
    height: "44px"
  button-secondary-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  button-ghost:
    textColor: "{colors.ink}"
    typography: "{typography.control}"
    rounded: "{rounded.none}"
    padding: "0 4px"
    height: "44px"
  button-ghost-hover:
    textColor: "{colors.live}"
  button-large:
    padding: "0 24px"
    height: "52px"
  button-disabled:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink-3}"
  input:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "8px 12px"
    height: "44px"
  thumb-tab:
    backgroundColor: "{colors.live}"
    textColor: "{colors.paper}"
    rounded: "{rounded.tab}"
    width: "30px"
    height: "84px"
  thumb-tab-current:
    width: "38px"
  thumb-tab-empty:
    backgroundColor: "{colors.hair}"
  step-active:
    backgroundColor: "{colors.live-wash}"
    textColor: "{colors.live}"
  step-failed:
    textColor: "{colors.fail}"
  badge:
    textColor: "{colors.ink-3}"
    typography: "{typography.label}"
  citation-mark:
    textColor: "{colors.ink}"
---

# Design System: infer

## Overview

**Creative North Star: "The Dictionary of Inferences"**

infer is a pocket dictionary printed on bible paper. The wordmark is the headword "infer, v.". Each model is an entry that a visitor can run. Each limit of the phone is a usage note. Black ink sits on bright paper, and the reverse page shows through, faint and mirrored. One ultramarine marks only what runs now.

The page reads like a printed reference. A running head with guide words sits over a ruled line. Heavy, tight grotesk headwords open each entry. Numbered senses follow in a text serif. Thumb tabs are cut into the fore-edge on the right. Numbers carry small citation marks that point to the run that produced them. Density is high and the voice is plain. Structure comes from rules and type weight, never from boxes, cards or shadows.

Motion is a stepped roll. When a line of a run changes state, it snaps up by one line with a small overshoot. A failed line jams half a line low. The world refuses the dark AI launch page with a glowing device mock and three feature cards.

**Key Characteristics:**
- Bright bible paper, black ink, one live ultramarine, one failure red.
- Two voices: Archivo for headwords, numbers and controls, Source Serif 4 for every sentence.
- Ruled lines and square corners. No cards and no drop shadows.
- The reverse page shows through as real mirrored text in columns.
- Blue thumb tabs on the fore-edge, one per entry that runs.
- Every measured number carries an ink citation mark.
- Whole-line snap with overshoot, and a half-line drop on failure.

## Colors

Ink on paper, with one live blue and one red, each held to a single meaning.

### Primary
- **Live Ultramarine** (`live`): the run that goes on now. It fills the primary button and the active step line. The active line also gets a light wash (`live-wash`). The progress band is blue while it loads, and so is the memory meter. Every thumb tab of an entry that runs is blue. Blue also answers the hand: link hover, focus outline, text caret and selection.

### Tertiary
- **Failure Red** (`fail`): the color of failure. It marks a failed step, a failed band, a danger meter and a danger log line. It also sets the notice rule. Nothing else is red.

### Neutral
- **Bible Paper** (`paper`): the page ground. It sits on the html element so the reverse page can show through below the body.
- **Press Black** (`ink`): text, headwords, the 1.5px structural rules, button borders, the ink button, the hover of the primary button, and citation marks.
- **Second Ink** (`ink-2`): secondary prose, step details, guide links, citations and the colophon.
- **Faded Ink** (`ink-3`): pending steps, placeholders, hints, idle badges and disabled controls.
- **Hairline** (`hair`): 1px row dividers, the empty progress track, meter ticks, the empty thumb cut and the column rule of a spread.

### Named Rules
**The Live Ink Rule.** Blue marks what runs now, or what the hand is on. A static fact, a heading, a done step or a citation is never blue.

**The Run Button Rule.** Only an action that starts a run gets the blue primary button. A main action that runs nothing, such as storing the weights, is the black ink button.

**The Ink Citation Rule.** Citation marks are ink. They point to a source, they do not run.

**The One Red Rule.** Red means failure and only failure. A warning is set in second ink and italic, never in red or amber.

## Typography

**Display Font:** Archivo Variable (with Helvetica Neue, Arial)

**Body Font:** Source Serif 4 Variable (with Georgia, Times New Roman)

**Label/Mono Font:** Archivo at 80% width for labels. ui-monospace (SF Mono, Menlo) for raw engine output only.

**Character:** A heavy, tight grotesk speaks the headwords and a quiet optical-size serif speaks the senses, as in a printed dictionary. The grotesk's width axis runs condensed for guide words and labels.

### Hierarchy
- **Display** (900, the `display` clamp, 0.8): the home headword "infer". On a screen of 980px or wider the home opening grows it to clamp(14rem, 27vw, 24rem) beside its senses.
- **Headline** (900, the `headline` clamp, 0.88): the headword of a model entry, such as "sd-turbo".
- **Title** (850, 1.25rem, 0.92): a test name on a developer page. All h1 to h3 share weight 850, tracking -0.035em and balanced wrapping.
- **Term** (850, 1.0625rem): the bold word that opens a usage note, a step title, a sense number.
- **Sense** (400, 1.1875rem, 1.42): the numbered senses of an opening.
- **Body** (400, 1.0625rem, 1.45): every sentence. Old-style proportional figures. Measure 68ch.
- **Label** (750, 0.875rem, 0.02em, 80% width, lowercase): section labels over a rule, such as "contents", "usage notes", "sources". The running head uses 700 at the same size and width.
- **Number** (Archivo, lining tabular figures, 92% width): any number that lines up in a column, sizes in the contents map, times in steps and progress.
- **Citation** (400, 0.75rem, 1.45): the source list and the colophon.
- **Data** (400, 0.75rem, 1.55, monospace): engine logs only.

### Named Rules
**The Two Voices Rule.** The grotesk sets headwords, numbers, labels and controls. The serif sets every sentence a visitor reads. Monospace appears only for raw engine lines.

**The Part of Speech Rule.** Italic marks the part of speech ("v.", "n. text → image"). The build also uses serif italic for two provisional voices, the empty plate before an image lands and the warning badge. Italic goes no further.

**The Even Figures Rule.** Prose numbers are old-style. Numbers in a column are lining and tabular, in the grotesk.

## Layout

The page is one column of a book. It is at most 1180px wide and centred. The gutter on the left is 18px, and 40px from 760px. The fore-edge on the right is 46px, and 72px from 760px. It leaves room for the thumb tabs. Space steps on a 4px base: 4, 8, 12, 16, 24, 32, 48, 72.

The running head is sticky at the top on paper. "infer" sits on the left and links home. The guide word and the links to the entry's other pages sit on the right. A 1.5px ink rule runs under it.

On a phone everything stacks in one column at 390px. From 980px the home page opens as a spread. The headword stands on the left and its senses stand at its foot on the right. Below, entries sit on the left and usage notes on the right. A hair rule splits them. The entry page splits into controls and the run on the left and the figure on the right. The foot of a page sits over a 1.5px ink rule, with sources and colophon in two columns from 760px.

Breakpoints are 760px (gutters, thumb size, third show-through column, footer columns) and 980px (spreads).

## Elevation & Depth

The page is flat paper. Depth comes from the paper itself: the reverse page shows through. A fixed layer under the page holds real dictionary text from the site. The text is mirrored left to right at 0.028 opacity. It runs in two grid columns on a phone and three from 760px. Each column is its own block of justified, hyphenated text cut at the foot of the screen. The layer never takes a tap and screen readers skip it.

The only box-shadow is the inset 1px live ring that thickens a focused input to a 2.5px blue border. The running head and thumb tabs lie above the page by z-index, not by shadow.

### Shadow Vocabulary
- **Focus thickening** (`box-shadow: inset 0 0 0 1px var(--live)`): a focused input only.

### Named Rules
**The Thin Paper Rule.** Depth is the show-through of real text, mirrored and faint. No drop shadows, no blur, no glow, no lifted cards.

## Shapes

Every corner is square (0px): buttons, inputs, plates, meters, bands. Structure is drawn with rules. A 1.5px ink rule opens a section, underlines the running head and closes the foot. A 1px hair rule divides rows of steps, logs and tests. The one curve is the thumb tab, cut with a 5px radius on its two left corners only, as a die cuts a fore-edge tab.

**The Ruled Line Rule.** A 1.5px ink rule marks structure. A 1px hair marks a row. Nothing is boxed in a card.

**The Square Corner Rule.** Corners are square. The thumb tab's cut is the single exception.

## Components

### Buttons
Square, ruled and set in the heavy grotesk, 44px high for a thumb.
- **Shape:** square corners (0px), 1.5px border, 44px high, 16px side padding. Large is 52px high with 24px padding at 1.1875rem.
- **Primary:** live blue fill, paper text. It starts a run: "run", "run all", "start". Hover turns it black.
- **Ink:** black fill, paper text. The main action of a step that runs nothing, such as "store 2.24 GB on this device". Hover lightens to second ink.
- **Secondary:** ink outline on paper. Hover fills it black. For "stop", "reset", per-test "run", exports.
- **Ghost:** a plain underlined word (1.5px underline), weight 600. Hover turns it blue. For "new seed", "save image", "copy log".
- **Disabled:** paper fill, faded ink text, hair border.
- **Focus:** a 2px live outline at 3px offset. Color changes take 160ms.

### Inputs / Fields
- **Style:** a ruled box, 1.5px ink border, square, paper fill, serif text at 1.0625rem so Safari does not zoom. The label sits above in the grotesk (600, 0.875rem) with an optional serif hint in faded ink.
- **Focus:** the border turns blue and thickens with an inset ring. The caret is blue.
- **Run line:** the prompt field and the run button join into one ruled line, the field losing its right border.

### Navigation
- **Running head:** guide words in condensed grotesk over a 1.5px rule, sticky. The current entry page link is ink and underlined. The others are second ink at weight 500. A new guide word fades in over 120ms.
- **Thumb index:** blue tabs fixed on the right edge, 112px from the top. A tab is 30 by 84px, and 44 by 112px from 760px. It carries the entry's letter in paper at weight 850. Hover pushes a tab out 6px. The open entry's tab sticks out further (38px, 54px wide). One hair-grey empty cut follows the last tab. It holds no link.
- **Cross-reference:** "→ word" in the grotesk at weight 700, as a dictionary prints "see". The arrow is a typographic sign in the line, not an icon.

### Badges
One lowercase grotesk word (750, 0.75rem, 0.04em). Idle is faded ink, success is ink, running is blue, failure is red, warning is second ink in italic.

### Steps (signature)
The numbered lines of a run. Each line is a grid of number, title over detail, and time, between hair rules. Pending lines are faded ink. The active line is blue on the live wash. A done line is ink with its time. A failed line is red. When a line changes state it snaps in: it rises from 1.1em below over 420ms on the `--snap` curve, passes its place a little and settles. A failed title stays 0.3em low, half a line short, like a jammed roll.

### Contents map, band and meter
- **Contents map:** a three-column grotesk grid under a hair rule: bold part number, part name, true size right-aligned in tabular figures.
- **Band:** a 4px hair track under a grotesk label, with the amount on the right. The fill is blue during the load. It turns ink when the load is done and red on a failure. An unknown length shuttles a 24% blue segment.
- **Meter:** a 10px ink-framed scale with hair ticks every tenth. Blue fills it to the live amount. The fill turns red over budget.

### Figure and citations
- **Plate:** a square with a 1px ink frame on paper. It shows an italic serif placeholder until the image covers it edge to edge. The caption runs under it as "fig. 1" in the grotesk, then serif.
- **Citation mark:** a raised ink numeral in the grotesk (700, 0.62em) after a number. It links to the numbered serif source list at the foot.

### Notice and log
- **Notice:** a sentence in red under a 1.5px red rule, for a reason the page cannot go on.
- **Log:** engine lines in monospace between hair rules, at most 420px high and scrolling. Danger lines red, success lines bold ink, muted lines faded.

## Do's and Don'ts

### Do:
- **Do** keep blue for what runs now and for the hand on a control. Make an action that starts a run the blue primary button and a main action that runs nothing the black ink button.
- **Do** give every entry that runs a blue thumb tab, and end the index with one empty hair-grey cut.
- **Do** carry an ink citation mark on every measured number and list its source run at the foot.
- **Do** draw structure with a 1.5px ink rule and rows with a 1px hair, on square corners.
- **Do** set headwords in Archivo 900 with tight negative tracking, and every sentence in Source Serif 4.
- **Do** use lining tabular grotesk figures for numbers in columns, and old-style figures in prose.
- **Do** step a state change one whole line on `--snap`, and leave a failed line half a line low.
- **Do** keep text 16px or larger in inputs and controls 44px high.

### Do not:
- **Do not** use blue for decoration, headings, done states or citation marks.
- **Do not** use red for anything but failure, and do not add an amber or green.
- **Do not** add drop shadows, glows, blurs, gradients or cards. Depth is the show-through.
- **Do not** round corners other than the thumb tab's cut.
- **Do not** set prose in monospace. Monospace is for raw engine lines.
- **Do not** set a dark theme. The world is bright paper with black ink.
- **Do not** raise the show-through above a faint 0.028, or fill it with placeholder text. It is real text of the book.
- **Do not** add a pictogram icon. A cross-reference is the typographic "→" in the line.
