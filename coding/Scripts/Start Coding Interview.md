<%*
// "Start coding this interview"
// Run while an interview from 01_Raw_Transcripts is open.
// Copies it into 02_Coding_In_Progress with the SAME file name (no " 1" suffix)
// and opens the copy. If the copy already exists, it just opens it.
// The raw file itself is never changed.

const RAW = "01_Raw_Transcripts";
const WIP = "02_Coding_In_Progress";

const src = tp.config.target_file;

if (!src || !src.path.startsWith(RAW + "/")) {
    new Notice(`Open an interview from ${RAW} first, then run this command.`);
} else {
    const destPath = `${WIP}/${src.name}`;
    let dest = app.vault.getAbstractFileByPath(destPath);

    if (dest) {
        new Notice(`${src.name} is already in ${WIP} – opening it.`);
    } else {
        if (!app.vault.getAbstractFileByPath(WIP)) {
            await app.vault.createFolder(WIP);
        }
        dest = await app.vault.copy(src, destPath);
        new Notice(`Started coding: ${src.name}`);
    }

    await app.workspace.getLeaf(false).openFile(dest);
}
-%>
