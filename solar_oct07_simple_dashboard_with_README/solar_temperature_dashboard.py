#!/usr/bin/env python3
"""Solar inverter day-wise thermal comparison dashboard.

Usage: python solar_temperature_dashboard.py "ADHOC_7oct.csv" --output "Oct07_dashboard"

Outputs: index.html with individually colored deviation CELLS (not entire rows),
        overview.csv, per-inverter CSV files, and a compact daily_module_summary.csv.
No median calculations, no temperature rise/fall rates. This is screening, not an
approved protective trip algorithm. Actual temperature is retained for safety.
"""
import argparse
import csv
import html
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

TEMPS = {f'Heatsink Temperature {i} (DEGREE CELSIUS)':f't{i}' for i in range(1,7)}
FIELDS = {**TEMPS, 'DC Current (A)':'current', 'Power Factor (AU)':'pf', 'Active Power (kW)':'power'}
PATTERN = re.compile(r'\b(B\d+-INV\d+-M\d+)\s*-\s*(.*?)\s*$')

def n(s):
    try:
        z=float(s)
        return z if math.isfinite(z) else None
    except (ValueError,TypeError): return None

def avg(seq):
    x=[v for v in seq if v is not None]
    return statistics.fmean(x) if x else None

def fmt(x, digits=2):
    return '' if x is None else f'{x:.{digits}f}'

def key_module(module):
    return int(module.rsplit('M',1)[-1])

def read_input(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        r=csv.reader(f); headers=next(r)
        if 'DATETIME' not in headers: raise ValueError('Missing DATETIME header')
        timestamp_col=headers.index('DATETIME')
        columns=[]; modules=set()
        for j,h in enumerate(headers):
            match=PATTERN.search(h)
            if match and match.group(2) in FIELDS:
                mod=match.group(1); columns.append((j,mod,FIELDS[match.group(2)])); modules.add(mod)
        if not columns: raise ValueError('No recognized inverter measurements found')
        data=[]
        for row in r:
            if len(row)<=timestamp_col or not row[timestamp_col].strip():continue
            timestamp=row[timestamp_col].strip()
            bymod={m:{'timestamp':timestamp,'module':m,'inverter':m.rsplit('-M',1)[0]} for m in modules}
            for j,m,metric in columns:
                bymod[m][metric]=n(row[j]) if j<len(row) else None
            for item in bymod.values():
                values=[item.get('t'+str(i)) for i in range(1,7)]
                values=[x for x in values if x is not None]
                item['avg']=avg(values);item['max']=max(values) if values else None
                item['sensor_count']=len(values)
                data.append(item)
    if not data:raise ValueError('No readings found')
    return data

def slope(pairs):
    if len(pairs)<12:return 0.0
    xm=avg([x for x,y in pairs]);ym=avg([y for x,y in pairs])
    den=sum((x-xm)**2 for x,y in pairs)
    return sum((x-xm)*(y-ym) for x,y in pairs)/den if den>1e-9 else 0.0

def compute(data,min_current):
    """Within-module de-meaned current adjustment fitted separately for each inverter.
    PF is not forced into normalization; daily PF barely varies and may be absent.
    """
    inv=defaultdict(list); groups=defaultdict(list)
    for r in data:
        inv[r['inverter']].append(r)
        groups[(r['timestamp'],r['inverter'])].append(r)
    models={}
    for inverter,rr in inv.items():
        valid=[r for r in rr if r.get('current') is not None and r['current']>=min_current and r['avg'] is not None]
        iref=avg([r['current'] for r in valid])
        models[inverter]={'iref':iref}
        for metric in ('avg','max'):
            bymod=defaultdict(list)
            for r in valid:
                if r[metric] is not None: bymod[r['module']].append(r)
            centered=[]
            for readings in bymod.values():
                ic=avg([r['current'] for r in readings]);tc=avg([r[metric] for r in readings])
                centered.extend((r['current']-ic,r[metric]-tc) for r in readings)
            models[inverter][metric+'_ki']=slope(centered)
    for r in data:
        model=models[r['inverter']];current=r.get('current')
        active=current is not None and current>=min_current and model['iref'] is not None
        for metric in ('avg','max'):
            r[metric+'_normalized']=(r[metric]-model[metric+'_ki']*(current-model['iref'])
                                     if active and r[metric] is not None else None)
    for group in groups.values():
        for metric in ('avg_normalized','max_normalized'):
            valid=[r[metric] for r in group if r[metric] is not None]
            minimum=min(valid) if len(valid)>=2 else None
            for r in group:
                r['dev_'+metric]=max(0.,r[metric]-minimum) if minimum is not None and r[metric] is not None else None
    return inv,models

def export(path,columns,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)

def threshold_class(value,threshold):
    if value is None:return ''
    if value>=threshold:return 'danger'
    if value>=threshold*.65:return 'watch'
    return ''

def build(data, out, threshold, warn_temp, min_current):
    byinv,models=compute(data,min_current)
    out.mkdir(parents=True,exist_ok=True); per=out/'inverters';per.mkdir(exist_ok=True)
    summary=[];overview=[];html_sections=[]
    for inverter,rr in sorted(byinv.items()):
        mods=sorted({r['module'] for r in rr},key=key_module)
        stamps=sorted({r['timestamp'] for r in rr})
        records={(r['timestamp'],r['module']):r for r in rr}
        time_headers=['Timestamp']
        for m in mods:
            suffix=m.rsplit('-M',1)[-1]
            time_headers += [f'M{suffix} Avg °C',f'M{suffix} Max °C',f'M{suffix} ΔAvg °C',f'M{suffix} ΔMax °C']
        output_rows=[]; flagged=set(); high_cells=0
        cells=[]
        for stamp in stamps:
            output_row={'Timestamp':stamp};tds=[f'<td class="stamp">{html.escape(stamp)}</td>']
            for mod in mods:
                r=records.get((stamp,mod)); suffix=mod.rsplit('-M',1)[-1]
                if r is None:r={}
                descriptors=[('avg','Avg °C'),('max','Max °C'),('dev_avg_normalized','ΔAvg °C'),('dev_max_normalized','ΔMax °C')]
                for field,label in descriptors:
                    value=r.get(field);output_row[f'M{suffix} {label}']=fmt(value)
                    klass=''
                    if field.startswith('dev_'):
                        klass=threshold_class(value,threshold)
                        if klass=='danger':flagged.add(mod);high_cells+=1
                    elif field=='max' and warn_temp is not None and value is not None and value>=warn_temp:
                        klass='absolute';flagged.add(mod);high_cells+=1
                    title=html.escape(f'{mod} | {field} | {stamp} | current={fmt(r.get("current"))} A | PF={fmt(r.get("pf"),4)} | power={fmt(r.get("power"))} kW')
                    t=f'{fmt(value)}' if value is not None else '—'
                    tds.append(f'<td class="{klass}" title="{title}">{t}</td>')
            output_rows.append(output_row); cells.append('<tr>'+''.join(tds)+'</tr>')
        export(per/(inverter+'.csv'),time_headers,output_rows)
        overview.append({'Inverter':inverter,'Flagged modules':', '.join(x.split('-')[-1] for x in sorted(flagged,key=key_module)),
                         'Flagged cells':high_cells,'Peak normalized ΔMax °C':fmt(max((r.get('dev_max_normalized') for r in rr if r.get('dev_max_normalized') is not None),default=None)),
                         'Peak actual temperature °C':fmt(max((r.get('max') for r in rr if r.get('max') is not None),default=None))})
        for m in mods:
            xs=[r for r in rr if r['module']==m]
            peak=max((r for r in xs if r['max'] is not None),key=lambda r:r['max'],default=None)
            devs=[r['dev_max_normalized'] for r in xs if r['dev_max_normalized'] is not None]
            summary.append({'Date':stamps[0][:10], 'Inverter':inverter,'Module':m,
                            'Daily average °C':fmt(avg([r['avg'] for r in xs])),
                            'Daily max °C':fmt(peak['max'] if peak else None),
                            'Timestamp of max':peak['timestamp'] if peak else '',
                            'Current at max A':fmt(peak.get('current') if peak else None),
                            'PF at max':fmt(peak.get('pf') if peak else None,4),
                            'Power at max kW':fmt(peak.get('power') if peak else None),
                            'Maximum normalized deviation °C':fmt(max(devs) if devs else None),
                            'Flagged ΔMax timestamps':sum(r['dev_max_normalized'] is not None and r['dev_max_normalized']>=threshold for r in xs)})
        heads='<th>Timestamp</th>'+''.join(f'<th>{html.escape(h)}</th>' for h in time_headers[1:])
        html_sections.append(f'<section class="inverter" id="{inverter}"><h2>{inverter} <small>{len(mods)} modules · {len(stamps)} times · {high_cells} highlighted cells</small></h2>'
          f'<p><a href="inverters/{inverter}.csv">Download this inverter CSV</a> · Hover a cell for current, PF and power.</p>'
          f'<div class="tablewrap"><table><thead><tr>{heads}</tr></thead><tbody>{"".join(cells)}</tbody></table></div></section>')
    export(out/'inverter_overview.csv',list(overview[0]),overview)
    export(out/'daily_module_summary.csv',list(summary[0]),summary)
    overview_table=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in overview[0])+'</tr>' for r in sorted(overview,key=lambda x:float(x['Peak normalized ΔMax °C'] or -1),reverse=True))
    choices=''.join(f'<option value="{html.escape(i)}">{html.escape(i)}</option>' for i in sorted(byinv))
    css='''*{box-sizing:border-box}body{margin:0;font-family:system-ui,Arial,sans-serif;background:#f3f6fa;color:#172b45}header{padding:24px 30px;background:#133357;color:white}h1{margin:0 0 7px;font-size:25px}.hint{font-size:13px;color:#cbdcf0}.wrap{padding:22px;max-width:1700px;margin:auto}nav{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px}select{padding:10px;border-radius:8px;border:1px solid #bccadd;font-size:15px}.panel{background:white;border:1px solid #dce3ed;padding:18px;border-radius:12px;margin-bottom:18px}h2{margin:0 0 6px;font-size:20px}small{color:#6b7787;font-size:12px;font-weight:normal}.tablewrap{max-height:670px;overflow:auto;border:1px solid #e4e9f1;border-radius:8px}table{border-collapse:separate;border-spacing:0;font-variant-numeric:tabular-nums;font-size:12px}th{background:#194570;color:white;position:sticky;top:0;z-index:2;text-align:center}th,td{border-right:1px solid #e6eaf0;border-bottom:1px solid #e6eaf0;min-width:80px;padding:7px;text-align:right;white-space:nowrap}th:first-child,td:first-child{position:sticky;left:0;z-index:3;min-width:155px;text-align:left;background:#eef3fa}th:first-child{z-index:5;background:#194570}.danger{background:#ffb6b6!important;color:#760e18;font-weight:700}.watch{background:#fff0c5!important;color:#895600;font-weight:600}.absolute{outline:2px solid #bf2444;outline-offset:-2px;background:#ffd6e1!important;font-weight:700}.inverter{display:none}.inverter.selected{display:block}.leg{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}.chip{padding:5px 8px;border-radius:5px}.note{color:#536477;font-size:13px;line-height:1.55}a{color:#1469ad}#overview{max-height:200px;overflow:auto}#overview td{min-width:130px}#overview th{min-width:130px}'''
    js='''const selector=document.getElementById('pick');function show(){document.querySelectorAll('.inverter').forEach(x=>x.classList.toggle('selected',x.id===selector.value));}selector.addEventListener('change',show);show();'''
    doc=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Solar Thermal Comparison</title><style>{css}</style></head><body><header><h1>Inverter thermal comparison · {html.escape(data[0]['timestamp'][:10])}</h1><div class="hint">Individual cell highlights only · Same inverter, same timestamp · Current-adjusted deviations from coolest module</div></header><div class="wrap"><div class="panel"><nav><strong>Select inverter</strong><select id="pick">{choices}</select><a href="inverter_overview.csv">Overview CSV</a><a href="daily_module_summary.csv">Daily summary CSV</a></nav><div class="leg"><span class="chip danger">Δ ≥ {threshold:g} °C — investigate</span><span class="chip watch">Δ ≥ {threshold*.65:g} °C — watch</span>{f'<span class="chip absolute">Actual max ≥ {warn_temp:g} °C</span>' if warn_temp is not None else ''}</div><p class="note">Each module has only 4 columns: average, maximum, normalized deviation of average (ΔAvg) and normalized deviation of maximum (ΔMax). Zero deviation = coolest module. Blank deviation = not evaluable (low current, missing data, or fewer than two modules). Temperature normalization uses a daily fitted current coefficient. PF is displayed in cell tooltips but <b>not used to adjust temperature</b> because it cannot be reliably estimated from this single day. Red means a screening flag, not a diagnosed fault or approved trip setting.</p></div><div class="panel"><h2>Inverter overview</h2><div class="tablewrap" id="overview"><table><thead><tr>{''.join('<th>'+html.escape(h)+'</th>' for h in overview[0])}</tr></thead><tbody>{overview_table}</tbody></table></div></div>{''.join(html_sections)}</div><script>{js}</script></body></html>'''
    (out/'index.html').write_text(doc,encoding='utf-8')
    print(f'OK: {len(data)} module readings | {len(byinv)} inverters | {len(summary)} modules | {len(overview)} inverter tables')
    print('Dashboard:',(out/'index.html').resolve())

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path)
    ap.add_argument('--output',type=Path,default=Path('thermal_dashboard'))
    ap.add_argument('--warn-deviation',type=float,default=8.0)
    ap.add_argument('--min-current',type=float,default=20.0)
    ap.add_argument('--warn-temp',type=float,default=None,help='Manufacturer-approved max temperature warning in °C (optional)')
    args=ap.parse_args()
    if args.warn_deviation<=0:ap.error('--warn-deviation must be positive')
    build(read_input(args.input),args.output,args.warn_deviation,args.warn_temp,args.min_current)
