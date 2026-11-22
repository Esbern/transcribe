<%*
// "Remove old interview audio"
// Permanently deletes every audio file in the vault that is NOT linked from the
// interview currently in 02_Coding_In_Progress. After switching to the next
// interview branch, the previous recording disappears from this computer.
//
// Set it as a Templater "Startup template" so it runs each time Obsidian opens;
// it can also be run by hand at any time.
// Safety: if 02_Coding_In_Progress has no notes, nothing is deleted.

const WIP = "02_Coding_In_Progress";
const AUDIO_EXT = ["mp3", "m4a", "wav", "flac", "ogg", "webm", "mp4"];

const workNotes = app.vault.getMarkdownFiles().filter(f => f.path.startsWith(WIP + "/"));

if (workNotes.length === 0) {
    new Notice(`No interview in ${WIP} – no audio removed.`);
} else {
    // Collect every audio file name mentioned in the working transcript(s).
    let text = "";
    for (const note of workNotes) text += "\n" + await app.vault.cachedRead(note);
    try { text += "\n" + decodeURIComponent(text); } catch (e) { /* keep raw text */ }

    const audioFiles = app.vault.getFiles().filter(f => AUDIO_EXT.includes(f.extension.toLowerCase()));
    const toDelete = audioFiles.filter(f => !text.includes(f.name));

    for (const f of toDelete) {
        await app.vault.delete(f, true); // permanent delete, not the Obsidian trash
    }

    const kept = audioFiles.length - toDelete.length;
    if (toDelete.length > 0) {
        new Notice(`Removed ${toDelete.length} old recording(s). Kept ${kept}.`);
    } else if (kept === 0) {
        new Notice("No recording for this interview yet – download it from SharePoint.");
    }
}
-%>
