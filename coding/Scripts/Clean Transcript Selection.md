<%*
// Cleans up a selected chunk of Whisper-transcribed interview text:
// - drops "### SPEAKER_XX [00:00:00](...)" style header lines (bold or not)
// - drops AI "Validation Issues" boxes (lines starting with ">")
// - drops empty lines
// - joins the remaining content lines into one paragraph, separated by spaces

let text = tp.file.selection();
let lines = text.split("\n");

let headerPattern = /^#{1,6}\s.*\[\d{1,2}:\d{2}:\d{2}\]\([^)]*#t=\d+\)\s*$/;
// AI "Validation Issues" callout boxes (lines starting with ">")
let calloutPattern = /^>/;

let kept = lines
    .filter(line => line.trim() !== "")
    .filter(line => !headerPattern.test(line.trim()))
    .filter(line => !calloutPattern.test(line.trim()))
    .map(line => line.trim());

let merged = kept.join(" ").replace(/\s+/g, " ").trim();

tR += merged;
-%>
