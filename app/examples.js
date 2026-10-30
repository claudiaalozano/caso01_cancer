"use strict";

const cortesEjemplo = [
  {
    paciente: "ISPY1_1001",
    corte: "017"
  },
  {
    paciente: "ISPY1_1001",
    corte: "018"
  },
  {
    paciente: "ISPY2-756412",
    corte: "014"
  },
  {
    paciente: "ISPY2-756412",
    corte: "015"
  },
  {
    paciente: "Breast_MRI_005",
    corte: "025"
  },
  {
    paciente: "Breast_MRI_005",
    corte: "026"
  }
];

window.DCE_EXAMPLES = cortesEjemplo.map(({ paciente, corte }) => {
  const base = `examples/${paciente}_z${corte}`;

  return {
    label: `${paciente} · corte ${corte}`,
    paths: {
      PRE: `${base}_PRE.png`,
      EARLY: `${base}_EARLY.png`,
      LATE: `${base}_LATE.png`
    }
  };
});