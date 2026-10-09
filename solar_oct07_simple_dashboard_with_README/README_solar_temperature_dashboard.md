# Solar Inverter Temperature Dashboard — User Guide

## 1. Purpose

This Python script analyzes **one day's CSV export** of solar inverter modules. It compares the modules **within the same inverter at the same timestamp** to identify modules that are hotter than the coolest comparable module. It generates an easy-to-browse dashboard and compact CSV reports.

**No median and no temperature rise/fall-rate calculation are performed.** Highlighting is applied to **individual cells**, not whole rows or whole files. The results indicate candidates for investigation, not proven faults or an approved inverter protection system.

## 2. Requirements

- Python 3.9 or newer (recommended).
- No third-party packages are required: only Python's standard library is used.
- Input: CSV export with a `DATETIME` column and inverter/module measurement headers such as `B01-INV01-M01 - Heatsink Temperature 1 (DEGREE CELSIUS)`.
- Accepted measurement names: `Heatsink Temperature 1` through `6 (DEGREE CELSIUS)`, `DC Current (A)`, `Power Factor (AU)`, `Active Power (kW)`.
- Input should contain data for **one calendar day**; the script itself does not restrict rows to a date.

## 3. Quick start — Windows Command Prompt

1. Extract the ZIP to a convenient folder.
2. Place the day's CSV file in that folder, next to `solar_temperature_dashboard.py` (or use a complete path to the CSV).
3. Open Command Prompt in the folder and run:

```bat
python solar_temperature_dashboard.py "ADHOC_7oct.csv" --output "Results_7oct"
```

Replace `ADHOC_7oct.csv` with the **exact CSV file name**, including `.csv`. Use quotation marks around names or paths containing spaces.

If Windows recognizes `py` instead of `python`, use:

```bat
py solar_temperature_dashboard.py "ADHOC_7oct.csv" --output "Results_7oct"
```

For a different date:

```bat
python solar_temperature_dashboard.py "ADHOC_8oct.csv" --output "Results_8oct"
```

When finished, double-click **`Results_7oct/index.html`** to open the dashboard in your browser. Internet access is not required.

### Optional settings

```bat
python solar_temperature_dashboard.py "ADHOC_7oct.csv" --output "Results_7oct" --warn-deviation 5
```

This sets the high-deviation screening threshold to **5 °C** instead of the default **8 °C**. The amber watch threshold is automatically 65% of this setting.

```bat
python solar_temperature_dashboard.py "ADHOC_7oct.csv" --output "Results_7oct" --min-current 30
```

This raises the minimum DC current eligible for normalization from the default **20 A** to **30 A**.

```bat
python solar_temperature_dashboard.py "ADHOC_7oct.csv" --output "Results_7oct" --warn-temp 80
```

This separately outlines cells when an **actual maximum temperature** is at least 80 °C. **80 °C is only an example**: only specify a temperature threshold justified by the manufacturer, sensor location, and operating instructions. Omit `--warn-temp` if the safe limit is not known.

To view all options:

```bat
python solar_temperature_dashboard.py --help
```

## 4. What is calculated

At each timestamp and for each module, using all non-missing available heatsink temperature channels (for the October 7 sample, these are sensors 1, 3, and 5):

- **Average temperature:** `T_avg = mean(valid temperature sensors)`.
- **Maximum temperature:** `T_max = max(valid temperature sensors)`.
- **Current normalization:** separately for average and maximum temperature,

  `T_normalized = T_measured - k_I * (I - I_ref)`

  `I_ref` is the mean current of valid, qualifying operating points for that inverter on that day; `k_I` is fitted from that day's data using a within-module centered linear slope, separately for average and maximum temperatures. This is a statistical comparison adjustment, **not a calibrated thermal model**. It should be validated over multiple days and load conditions before relying on it.

- **Deviation from the lowest:** at a given timestamp, among the modules of the **same inverter** with valid normalized temperatures:

  `Delta_avg(module) = T_avg_normalized(module) - min(T_avg_normalized in inverter)`

  `Delta_max(module) = T_max_normalized(module) - min(T_max_normalized in inverter)`

  The coolest valid module has deviation **0 °C**; larger positive values indicate relative excess temperature. A deviation is left blank if the comparison is unavailable (for example, insufficient valid modules, missing data, or low current).

**Power factor (PF) and active power:** retained as context in the dashboard cell hover text and daily summary. **PF is not used in the current normalization formula** because a reliable independent PF coefficient was not established from the single-day data. Do not describe the output as PF-normalized.

**No median. No rise/fall rates.**

## 5. Output files and how to read them

For output folder `Results_7oct`:

| File | Meaning |
| --- | --- |
| `index.html` | Main dashboard. Select an inverter from the dropdown; its modules appear horizontally, timestamps vertically. |
| `inverter_overview.csv` | One row per inverter; flagged modules/cells, highest normalized maximum deviation, and peak actual temperature. Start here to prioritize inspection. |
| `daily_module_summary.csv` | One row per module; daily average/maximum, time of maximum, current/PF/power at maximum, largest normalized maximum deviation, count of flagged timestamps. |
| `inverters/B01-INV01.csv` (etc.) | One compact file per inverter. Each module has four columns: Avg °C, Max °C, ΔAvg °C, ΔMax °C. |

### Highlight colors in `index.html`

- **Red cell:** normalized deviation >= `--warn-deviation` (default 8 °C). Investigate the particular timestamp/module/metric.
- **Amber cell:** normalized deviation >= 65% of the red threshold, but below the red threshold.
- **Pink outline on a Max °C cell:** the **actual** maximum reaches the optional `--warn-temp` threshold, if supplied.
- **Uncolored cell:** not flagged by these rules; this does **not** establish that the equipment is safe.
- **Blank deviation / dash:** comparison is not evaluable. Do not interpret missing data as zero.

The module name appears across the columns, **not as repeated data rows**. Hover over a value in the HTML dashboard to see the module, timestamp, current, PF, and power associated with that reading.

## 6. Recommended review workflow

1. Open `index.html` and check the **Inverter overview**.
2. Select an inverter whose `Peak normalized ΔMax °C` is large or whose modules have red cells.
3. Locate the exact red/amber cell; check whether the deviation occurs repeatedly.
4. Inspect Avg °C and Max °C beside that deviation. Hover over the cells to check current, PF, and power.
5. Cross-check original measurements, sensor condition/placement, ambient temperature and cooling configuration before attributing a fault.
6. Keep a separate `Results_YYYY-MM-DD` folder for each date. Trend analysis across days can be built from the daily summaries later.

## 7. Important limitations and safety

- Current normalization is **linear and fitted from just that day**. Thermal dynamics depend on loading history, cooling, ambient temperature, and possibly nonlinear current-related losses. These are not fully modeled.
- Comparison to the lowest module is **relative**. If all modules overheat together, their deviations may stay small. For equipment protection, independently monitor **absolute actual maximum temperature** against manufacturer-approved limits.
- The cooler module is not necessarily a verified healthy baseline; differences in sensor placement, hardware, firmware, or operating state may affect comparison.
- Only available sensor channels are used. Missing temperature channels must not be silently treated as zero.
- Do not use this offline dashboard alone to control, trip, or protect an operating inverter.

## 8. Files included in the package

- `solar_temperature_dashboard.py` — reusable analysis program.
- `README_solar_temperature_dashboard.md` — this guide.
- `solar_oct07_simple_dashboard/` — pre-generated October 7 example reports and HTML dashboard.

The original raw CSV is not included in this package. To analyze another day, supply that day's export separately.
