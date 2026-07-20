# Digitized measured FRFs of the reference rig

Point coordinates digitized from Fig. 12 of Du, Liu, Dai, Long,
"Robust combined time delay control for milling chatter suppression of
flexible workpieces", Int. J. Mech. Sci. 274 (2024) 109257 (measured
curves; right-upper-corner impact/response point of the plate).

- `measured_frf_receptance.csv` - impact test receptance (Fig. 12a,
  y-axis in dB, reference as printed: 1 um/N), 89 points, 5-5002 Hz.
- `measured_frf_voltage_to_displacement.csv` - piezo swept-sine
  transfer function (Fig. 12b, y-axis in dB, reference as printed:
  0.01 um/V), 76 points, 14-5006 Hz.

Caveats inherent to plot digitization: sharp resonance tips (measured
damping ratios 0.17-0.56 % imply 3 dB widths of a few Hz) are
under-sampled, so peak AMPLITUDES are lower bounds; frequencies of
local maxima and off-resonance backbone levels are reliable.
