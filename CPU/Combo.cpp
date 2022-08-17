
//Rotate combination buffer by offset amount
//Currently supports combo buffers with maximum length 8
void adjustComboBuffer(int8_t * combo, int offset) {

  if (SIZE_COMBO_MULTI > 4)
	while (offset >= 10000) {
    offset-=10000;
    combo[4]++;
    if (SIZE_COMBO_MULTI > 5 && combo[4] >= COUNT_COMBO_SYMBOLS) {
      combo[4] -= COUNT_COMBO_SYMBOLS;
      combo[5]++;
      if (SIZE_COMBO_MULTI > 6 && combo[5] >= COUNT_COMBO_SYMBOLS) {
        combo[5] -= COUNT_COMBO_SYMBOLS;
        combo[6]++;
        if (SIZE_COMBO_MULTI > 7 && combo[6] >= COUNT_COMBO_SYMBOLS) {
          combo[6] -= COUNT_COMBO_SYMBOLS;
          combo[7]++;
          if (combo[7] >= COUNT_COMBO_SYMBOLS) {
            combo[7]=0;
          }
        }
      }
    }
  }

  if (SIZE_COMBO_MULTI > 3)
  while (offset >= 1000) {
    offset-=1000;
    combo[3]+=10;
    if (SIZE_COMBO_MULTI > 4 && combo[3] >= COUNT_COMBO_SYMBOLS) {
      combo[3] -= COUNT_COMBO_SYMBOLS;
      combo[4]++;
      if (SIZE_COMBO_MULTI > 5 && combo[4] >= COUNT_COMBO_SYMBOLS) {
        combo[4] -= COUNT_COMBO_SYMBOLS;
        combo[5]++;
        if (SIZE_COMBO_MULTI > 6 && combo[5] >= COUNT_COMBO_SYMBOLS) {
          combo[5] -= COUNT_COMBO_SYMBOLS;
          combo[6]++;
          if (SIZE_COMBO_MULTI > 7 && combo[6] >= COUNT_COMBO_SYMBOLS) {
            combo[6] -= COUNT_COMBO_SYMBOLS;
            combo[7]++;
            if (combo[7] >= COUNT_COMBO_SYMBOLS) {
              combo[7]=0;
            }
          }
        }
      }
    }
  }

  if (SIZE_COMBO_MULTI > 3)
  while (offset >= 100) {
    offset-=100;
    combo[3]++;
    if (SIZE_COMBO_MULTI > 4 && combo[3] >= COUNT_COMBO_SYMBOLS) {
      combo[3] -= COUNT_COMBO_SYMBOLS;
      combo[4]++;
      if (SIZE_COMBO_MULTI > 5 && combo[4] >= COUNT_COMBO_SYMBOLS) {
        combo[4] -= COUNT_COMBO_SYMBOLS;
        combo[5]++;
        if (SIZE_COMBO_MULTI > 6 && combo[5] >= COUNT_COMBO_SYMBOLS) {
          combo[5] -= COUNT_COMBO_SYMBOLS;
          combo[6]++;
          if (SIZE_COMBO_MULTI > 7 && combo[6] >= COUNT_COMBO_SYMBOLS) {
            combo[6] -= COUNT_COMBO_SYMBOLS;
            combo[7]++;
            if (combo[7] >= COUNT_COMBO_SYMBOLS) {
              combo[7]=0;
            }
          }
        }
      }
    }
  }

  if (SIZE_COMBO_MULTI > 2)
  while (offset >= 10) {
    offset-=10;
    combo[2]+=10;
    if (SIZE_COMBO_MULTI > 3 && combo[2] >= COUNT_COMBO_SYMBOLS) {
      combo[2] -= COUNT_COMBO_SYMBOLS;
      combo[3]++;
      if (SIZE_COMBO_MULTI > 4 && combo[3] >= COUNT_COMBO_SYMBOLS) {
        combo[3] -= COUNT_COMBO_SYMBOLS;
        combo[4]++;
        if (SIZE_COMBO_MULTI > 5 && combo[4] >= COUNT_COMBO_SYMBOLS) {
          combo[4] -= COUNT_COMBO_SYMBOLS;
          combo[5]++;
          if (SIZE_COMBO_MULTI > 6 && combo[5] >= COUNT_COMBO_SYMBOLS) {
            combo[5] -= COUNT_COMBO_SYMBOLS;
            combo[6]++;
            if (SIZE_COMBO_MULTI > 7 && combo[6] >= COUNT_COMBO_SYMBOLS) {
              combo[6] -= COUNT_COMBO_SYMBOLS;
              combo[7]++;
              if (combo[7] >= COUNT_COMBO_SYMBOLS) {
                combo[7]=0;
              }
            }
          }
        }
      }
    }
  }

  if (SIZE_COMBO_MULTI > 2)
  while (offset > 0) {
    offset--;
    combo[2]++;
    if (SIZE_COMBO_MULTI > 3 && combo[2] >= COUNT_COMBO_SYMBOLS) {
      combo[2] -= COUNT_COMBO_SYMBOLS;
      combo[3]++;
      if (SIZE_COMBO_MULTI > 4 && combo[3] >= COUNT_COMBO_SYMBOLS) {
        combo[3] -= COUNT_COMBO_SYMBOLS;
        combo[4]++;
        if (SIZE_COMBO_MULTI > 5 && combo[4] >= COUNT_COMBO_SYMBOLS) {
          combo[4] -= COUNT_COMBO_SYMBOLS;
          combo[5]++;
          if (SIZE_COMBO_MULTI > 6 && combo[5] >= COUNT_COMBO_SYMBOLS) {
            combo[5] -= COUNT_COMBO_SYMBOLS;
            combo[6]++;
            if (SIZE_COMBO_MULTI > 7 && combo[6] >= COUNT_COMBO_SYMBOLS) {
              combo[6] -= COUNT_COMBO_SYMBOLS;
              combo[7]++;
              if (combo[7] >= COUNT_COMBO_SYMBOLS) {
                combo[7]=0;
              }
            }
          }
        }
      }
    }
  }
}