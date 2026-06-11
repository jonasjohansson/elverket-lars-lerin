// psd-staggered-video-matte.jsx
//
// Imports a layered PSD into After Effects and applies a video as a luma
// matte to each layer with staggered timing — so the layers reveal (or
// dissolve out) in sequence, each one wiping with the video's pattern.
//
// Designed for the Lerin workflow: a PSD where each painted object is on
// its own layer, with the inpainted/clean background at the bottom. The
// matte video drives a textured dissolve front per layer.
//
// Run via File → Scripts → Run Script File…  (or drop into
// /Applications/Adobe After Effects YYYY/Scripts/ScriptUI Panels/ to
// access via the Window menu).

(function () {
  if (app.project === null) { alert('Open or create a project first.'); return; }
  app.beginUndoGroup('PSD Staggered Video Matte');

  try {
    // -------------------- file pickers --------------------
    var psdFile = File.openDialog('Choose the layered PSD', 'PSD:*.psd,All:*.*');
    if (!psdFile) { app.endUndoGroup(); return; }

    var matteFile = File.openDialog('Choose the matte (video preferred, image works too)',
      'Matte:*.mp4;*.mov;*.avi;*.mkv;*.png;*.jpg;*.jpeg,All:*.*');
    if (!matteFile) { app.endUndoGroup(); return; }

    // -------------------- settings dialog --------------------
    var dlg = new Window('dialog', 'Staggered Video Matte');
    dlg.alignChildren = 'fill';

    function addRow(label, defaultVal) {
      var g = dlg.add('group'); g.alignChildren = 'left';
      g.add('statictext', undefined, label).preferredSize.width = 180;
      var input = g.add('edittext', undefined, String(defaultVal));
      input.preferredSize.width = 80;
      return input;
    }
    var perLayerIn = addRow('Per-layer reveal duration (s):', '2.0');
    var staggerIn  = addRow('Stagger offset between layers (s):', '0.6');
    var directionGrp = dlg.add('group'); directionGrp.alignChildren = 'left';
    directionGrp.add('statictext', undefined, 'Matte direction:').preferredSize.width = 180;
    var dissolveOut = directionGrp.add('radiobutton', undefined, 'Layer disappears as matte brightens (LumaInverted)');
    var revealIn   = directionGrp.add('radiobutton', undefined, 'Layer appears as matte brightens (Luma)');
    dissolveOut.value = true;

    var btns = dlg.add('group'); btns.alignment = 'right';
    var ok = btns.add('button', undefined, 'Build', { name: 'ok' });
    btns.add('button', undefined, 'Cancel', { name: 'cancel' });
    if (dlg.show() !== 1) { app.endUndoGroup(); return; }

    var perLayerSec = parseFloat(perLayerIn.text);
    var staggerSec  = parseFloat(staggerIn.text);
    if (isNaN(perLayerSec) || perLayerSec <= 0) perLayerSec = 2.0;
    if (isNaN(staggerSec) || staggerSec < 0)    staggerSec  = 0.6;
    var matteType = dissolveOut.value ? TrackMatteType.LUMA_INVERTED : TrackMatteType.LUMA;

    // -------------------- import PSD as a comp with layers --------------------
    var psdOpts = new ImportOptions(psdFile);
    psdOpts.importAs = ImportAsType.COMP_CROPPED_LAYERS;
    var psdItem = app.project.importFile(psdOpts);

    var comp = null;
    if (psdItem instanceof CompItem) comp = psdItem;
    else if (psdItem instanceof FolderItem) {
      for (var i = 1; i <= psdItem.items.length; i++) {
        if (psdItem.items[i] instanceof CompItem) { comp = psdItem.items[i]; break; }
      }
    }
    if (!comp) { alert('Could not locate composition from PSD'); app.endUndoGroup(); return; }

    // -------------------- import matte footage --------------------
    var matteOpts = new ImportOptions(matteFile);
    var matteItem = app.project.importFile(matteOpts);

    // -------------------- compute timeline + duration --------------------
    var numLayers = comp.numLayers;
    if (numLayers === 0) { alert('Comp has no layers — nothing to do.'); app.endUndoGroup(); return; }
    var totalDuration = perLayerSec + (numLayers - 1) * staggerSec;
    comp.duration = Math.max(totalDuration, 1);
    comp.workAreaStart = 0;
    comp.workAreaDuration = comp.duration;

    // -------------------- snapshot original layers --------------------
    // We add matte layers above each original layer, which shifts indices.
    // Snapshot references first so iteration stays stable.
    var originalLayers = [];
    for (var k = 1; k <= comp.numLayers; k++) originalLayers.push(comp.layer(k));

    // -------------------- attach matte per layer --------------------
    // In AE, layer 1 is top-most. Bottom layer (highest index) is "behind".
    // For dissolve-out workflow we typically want the BOTTOM-most painted
    // layer to dissolve LAST (since it's the underlying paint). Adjust the
    // reveal order so layer-1 (top in PSD) reveals first.
    for (var li = 0; li < originalLayers.length; li++) {
      var contentLayer = originalLayers[li];

      // Add the matte source and move it directly above the content layer
      var matteLayer = comp.layers.add(matteItem);
      matteLayer.moveBefore(contentLayer);

      // Stagger: top-of-PSD (li=0) starts at t=0, each subsequent layer
      // offset by staggerSec.
      matteLayer.startTime = li * staggerSec;

      // Stretch the matte to fill the per-layer reveal duration
      var matteSrcDur = matteItem.duration > 0 ? matteItem.duration : perLayerSec;
      matteLayer.stretch = (perLayerSec / matteSrcDur) * 100; // %

      matteLayer.name = 'matte → ' + contentLayer.name;
      matteLayer.enabled = false; // mattes are invisible (provide alpha only)

      contentLayer.trackMatteType = matteType;
    }

    alert('Built ' + numLayers + ' staggered ' +
          (matteType === TrackMatteType.LUMA_INVERTED ? 'dissolve-out' : 'reveal-in') +
          ' transitions.\nComp duration: ' + comp.duration.toFixed(2) + 's.');
  } catch (e) {
    alert('Error: ' + e.toString() + '\n' + (e.line ? 'line ' + e.line : ''));
  }

  app.endUndoGroup();
})();
