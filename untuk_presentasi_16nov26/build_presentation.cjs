/* Presentation construction only. Every empirical value comes from evidence.json. */
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const ROOT = __dirname;
const E = JSON.parse(fs.readFileSync(path.join(ROOT, 'sources/evidence.json'), 'utf8'));
// Attach exact table-row locations in memory, without rewriting source evidence.
const sourcesByKey={primary:'primary_eight_strata.csv',discovery_e1:'discovery_e1_ci.csv',
 discovery_e1_sensitivity:'discovery_e1_sensitivity.csv',discovery_auc:'discovery_auroc_per_seed.csv',
 amplitude:'discovery_amplitude_ci.csv',bridge:'discovery_bridge_associations.csv',
 historical_auc:'historical_auroc.csv',historical_auc_contrast:'historical_auroc_contrast.csv',
 calibration:'calibration_scores.csv',calibration_contrasts:'calibration_contrasts.csv'};
for(const [key,file] of Object.entries(sourcesByKey))E[key].forEach((r,i)=>{r._source_copy='sources/'+file;r._row_index=i;});
for(const [cohort,group] of Object.entries(E.replication))for(const [key,rows] of Object.entries(group)){
 if(Array.isArray(rows))rows.forEach((r,i)=>{r._source_copy=`sources/${cohort}_${key}.csv`;r._row_index=i;});
}
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Agriby Diandra Chaniago';
pptx.subject = 'Audit transfer ranking attention pada model tulisan tangan Parkinsonian';
pptx.title = 'Beda Peta, Masih Relevan?';
pptx.company = 'Universitas Harapan Bangsa';
pptx.lang = 'id-ID';
pptx.theme = {headFontFace:'Arial', bodyFontFace:'Arial', lang:'id-ID'};
const W=13.333333, H=7.5;
const C={ink:'18342F', text:'22362F', muted:'52645F', pale:'EDF4F1', white:'FFFFFF',
  own:'173D35', transfer:'248D77', donor:'117A74', median:'964F79', gold:'C48A3A',
  grid:'D7E2DD', light:'F6F8F7', zero:'6B7280', source:'6F4D86', warn:'8C4F2B'};
const slides=[], chartSpecs=[], shapeLog=[], displayBindings=[];
let chartNumber=0;
const fmt=(v,d=3)=>Number(v).toFixed(d).replace('-', '−');
const ci=(r,d=3)=>{
 if(!r._source_copy)throw Error('CI without a source binding');
 const rendered=`${fmt(r.estimate,d)} [${fmt(r.ci_low,d)}; ${fmt(r.ci_high,d)}]`;
 displayBindings.push({slide:slides.length,source_copy:r._source_copy,row_index:r._row_index,
  estimate_field:r._estimateField||'estimate',estimate:r.estimate,ci_low:r.ci_low,ci_high:r.ci_high,decimals:d,rendered});
 return rendered;
};
const model=a=>a==='gru'?'BiGRU':'BiMamba-2';
const masker=m=>m==='train_donor'?'Donor':'Median';
const condition=r=>`${model(r.architecture)} / ${masker(r.masker)}`;
function one(rows,predicate,context){const r=rows.filter(predicate);if(r.length!==1)throw Error(context+': '+r.length);return r[0];}
const primary=(cohort)=>E.primary.filter(r=>r.cohort===cohort);
function secondary(cohort,row,endpoint,scenario='all_primary'){
 return one(E.replication[cohort].secondary_mean_contrasts,r=>r.architecture===row.architecture&&r.masker===row.masker&&r.endpoint===endpoint&&r.scenario===scenario,'secondary '+endpoint);
}
function T(s,text,x,y,w,h,size=24,options={}){
 shapeLog.push({slide:s._presentationNumber,kind:'text',text:String(text).slice(0,160),x,y,w,h,size});
 s.addText(String(text),{x,y,w,h,fontFace:'Arial',fontSize:size,color:C.text,margin:0,
  breakLine:false,valign:'mid',paraSpaceAfterPt:0,...options});
}
function R(s,x,y,w,h,fill=C.pale,line=fill,radius=false){
 shapeLog.push({slide:s._presentationNumber,kind:'shape',x,y,w,h});
 s.addShape(radius?pptx.ShapeType.roundRect:pptx.ShapeType.rect,{x,y,w,h,
  rectRadius:0.15,fill:{color:fill},line:{color:line,width:0.6}});
}
function A(s,x1,y1,x2,y2,color=C.muted){
 s.addShape(pptx.ShapeType.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color,width:1.8,endArrowType:'triangle'}});
}
function pill(s,text,x,y,w,color=C.own){
 R(s,x,y,w,0.38,C.pale,C.pale,true);T(s,text,x+0.1,y+0.02,w-0.2,0.33,13,{color,bold:true});
}
function add(title,section,notes,source='',backup=false,dark=false){
 const concise={
  'Perubahan keluaran terlihat sebelum subset dipilih':'Keluaran antar-seed tidak selalu serupa',
  'Pada rekaman yang sama, keluaran serupa dapat disertai peta berbeda':'Keluaran serupa dapat disertai peta berbeda',
  'Bagaimana menilai perbedaan peta pada model yang sama?':'Bagaimana menilai peta berbeda pada model yang sama?',
  'Dua akuisisi menguji prosedur yang sama setelah retraining':'Audit direplikasi pada dua akuisisi spiral',
  'Bank 20 seed memperluas discovery, bukan menggantikan riwayatnya':'Bank 20 seed memperluas discovery',
  'Satu classifier menghasilkan probabilitas dan attention':'Classifier menghasilkan probabilitas dan attention',
  'Hanya sumber ranking yang berubah saat target diuji':'Target tetap ketika sumber ranking berubah',
  'Respons besar harus dibandingkan dengan intervensi yang sepadan':'Respons besar membutuhkan kontrol yang sepadan',
  'Keuntungan ranking dihitung terhadap kontrolnya sendiri':'Keuntungan ranking dihitung terhadap kontrol',
  'NewHandPD menunjukkan transfer relevansi secara parsial':'Relevansi berpindah sebagian pada NewHandPD',
  'UCI-SST mengulang urutan rata-rata yang sama':'UCI-SST mengulang pola rata-rata yang sama',
  'Matching menentukan pasangan yang diwakili hasil':'Matching menentukan cakupan hasil',
  'Perbedaan peta berkaitan dengan besar perbedaan relevansi':'JSD berkaitan dengan besar efek transfer',
  'Ranking nyata tidak selalu melampaui peta plasebo':'Ranking nyata tidak selalu melampaui plasebo',
  'Mencocokkan kontrol belum menyamakan dosis kedua ranking':'Dosis langsung kedua ranking masih dapat berbeda',
  'Rerata yang konsisten tetap menyisakan variasi target':'Pola rata-rata menyisakan variasi target',
  'Audit memberi tiga pemeriksaan untuk membandingkan penjelasan':'Tiga pemeriksaan untuk membandingkan penjelasan',
  'Bukti menjangkau dua akuisisi, dengan cakupan yang terdefinisi':'Cakupan bukti mengikuti akuisisi dan matching',
  'D · Bridge discovery mengestimasi hubungan yang berbeda':'D · Bridge discovery mengganti fungsi model',
  'E · Magnitude ranking juga dapat mengungguli attention':'E · Magnitude dapat mengungguli attention',
  'F · BiMamba-2 lebih konsisten secara deskriptif setelah retraining':'F · Konsistensi AUROC setelah retraining terpisah',
  'G · Discovery Brier menguntungkan BiMamba-2 pada kualitas probabilitas':'G · Brier dan ECE pada bank discovery',
  'H · Kontrol historis membatasi interpretasi diagnostik':'H · Usia dan baseline historis memberi konteks',
 };
 title=concise[title]||title;
 const s=pptx.addSlide();s._presentationNumber=slides.length+1;s.background={color:dark?C.ink:C.white};
 const meta={number:s._presentationNumber,title,section,notes,source,backup,dark};slides.push(meta);
 if(!dark){
  T(s,section.toUpperCase(),0.58,0.32,11.7,0.25,12,{color:C.transfer,bold:true,charSpacing:1.3});
  T(s,title,0.58,0.82,12.15,0.75,32,{color:C.ink,bold:true});
 }
 s.addNotes(notes+'\n\nSUMBER: '+source+'\nStatus: hasil tersimpan; tidak ada komputasi penelitian baru untuk presentasi.');
 return s;
}
function noteBox(s,text,x=0.65,y=6.3,w=12.05,h=0.48,color=C.muted){T(s,text,x,y,w,h,16,{color});}
function card(s,heading,body,x,y,w,h,fill=C.pale){
 R(s,x,y,w,h,fill,fill,true);T(s,heading,x+0.23,y+0.2,w-0.46,0.6,24,{bold:true,color:C.ink});
 T(s,body,x+0.23,y+0.96,w-0.46,h-1.1,20,{valign:'top',breakLine:false});
}
function containPng(file,x,y,w,h){
 const b=fs.readFileSync(file),pxW=b.readUInt32BE(16),pxH=b.readUInt32BE(20);
 const scale=Math.min(w/pxW,h/pxH),outW=pxW*scale,outH=pxH*scale;
 return {x:x+(w-outW)/2,y:y+(h-outH)/2,w:outW,h:outH};
}
function table(s,headers,rows,x,y,widths,rowH=0.52,font=18,headerH=0.56){
 let xx=x;headers.forEach((v,i)=>{R(s,xx,y,widths[i],headerH,C.ink,C.white);T(s,v,xx+0.1,y+0.04,widths[i]-0.2,headerH-0.08,font-1,{color:C.white,bold:true});xx+=widths[i];});
 rows.forEach((row,j)=>{let xx=x;row.forEach((v,i)=>{R(s,xx,y+headerH+j*rowH,widths[i],rowH,j%2?C.white:C.pale,C.white);T(s,v,xx+0.1,y+headerH+j*rowH+0.04,widths[i]-0.2,rowH-0.08,font,{valign:'mid'});xx+=widths[i];});});
}
function chart(s,type,data,opts={},errorRows=null){
 chartNumber++;
 const options={x:0.6,y:1.9,w:8.1,h:4.4,showTitle:false,showLegend:false,
  showValue:false,chartColors:[C.own,C.transfer,C.source,C.gold],
  catAxisLabelFontFace:'Arial',catAxisLabelFontSize:15,catAxisLabelColor:C.text,
  valAxisLabelFontFace:'Arial',valAxisLabelFontSize:14,valAxisLabelColor:C.muted,
  valGridLine:{color:C.grid,size:0.6},catGridLine:{style:'none'},
  catAxisLineShow:false,valAxisLineShow:false,showBorder:false,
  chartArea:{fill:{color:C.white},border:{color:C.white,pt:0}},
  plotArea:{fill:{color:C.white},border:{color:C.white,pt:0}},
  legendFontFace:'Arial',legendFontSize:15,legendColor:C.text,legendPos:'b',
  showCatName:false,showSerName:false,showCatAxisTitle:false,showValAxisTitle:true,
  valAxisTitle:'',valAxisTitleFontFace:'Arial',valAxisTitleFontSize:14,
  showMarker:false,lineDataSymbol:'none',...opts};
 s.addChart(type,data,options);
 chartSpecs.push({chart:`ppt/charts/chart${chartNumber}.xml`,slide:s._presentationNumber,
  series:data.map(x=>({name:x.name,labels:x.labels,values:x.values})),errors:errorRows,
  error_direction:options.barDir==='bar'?'x':'y',axis_title:options.valAxisTitle});
 return chartNumber;
}
function bars(s,rows,series,x,y,w,h,max,axisLabel,colors=null){
 const rev=[...rows].reverse();
 const data=series.map(q=>({name:q.name,labels:rev.map(q.label||condition),values:rev.map(r=>Number(q.get(r).estimate))}));
 const errors=series.map(q=>rev.map(r=>{const v=q.get(r);return {estimate:Number(v.estimate),low:Number(v.ci_low),high:Number(v.ci_high),source_copy:v._source_copy,row_index:v._row_index};}));
 return chart(s,pptx.ChartType.bar,data,{x,y,w,h,barDir:'bar',barGrouping:'clustered',
  barGapWidthPct:90,barOverlapPct:0,valAxisMinVal:0,valAxisMaxVal:max,
  valAxisTitle:axisLabel,showLegend:series.length>1,chartColors:colors||[C.own,C.transfer]},errors);
}

// 01 — The hook is a question about transferable relevance, not map failure.
{
 const s=add('Beda Peta, Masih Relevan?','Pembuka',
 'Presentasi ini bertanya apakah ranking attention dari seed pelatihan lain masih menunjuk bagian input yang berpengaruh pada model target. Kata relevan mempunyai arti operasional. Kami mengukurnya melalui respons terhadap intervensi yang terkontrol. Kami tidak menyamakan peta attention dengan lokasi gejala. Temuan utamanya adalah transfer parsial. Ranking sumber tetap mengungguli kontrol rata-rata, tetapi ranking milik target memberi keuntungan lebih besar. Semua perbandingan transfer berlangsung dalam arsitektur dan dataset yang sama.',
 'V9 Introduction, Methods, Results; sources/primary_eight_strata.csv',false,true);
 T(s,'Beda Peta,\nMasih Relevan?',0.68,0.82,8.1,1.85,50,{color:C.white,bold:true});
 T(s,'Audit transfer ranking attention\npada tulisan tangan Parkinsonian',0.72,3.0,7.5,1.05,26,{color:'D2E6DD'});
 T(s,'Agriby Diandra Chaniago',0.72,4.57,7.9,0.46,24,{color:C.white,bold:true});
 T(s,'Purwono · Annastasya Nabila Elsa Wulandari · Alfian Ma’arif',0.72,5.16,10.9,0.5,16,{color:'D2E6DD'});
 T(s,'Universitas Harapan Bangsa · Universitas Ahmad Dahlan',0.72,5.8,11.2,0.37,16,{color:'D2E6DD'});
 T(s,'16 November 2026',0.72,6.48,7.0,0.4,19,{color:C.white});
 const labels=['Peta dari seed lain','Model target tetap','Relevansi diuji'];
 labels.forEach((text,i)=>{R(s,9.1,1.12+i*1.48,3.5,0.97,'244B40','244B40',true);T(s,text,9.3,1.25+i*1.48,3.1,0.68,23,{color:C.white,bold:true});if(i<2)A(s,10.85,2.12+i*1.48,10.85,2.46+i*1.48,'9ED1BD');});
}

// 02 — Discrimination is not an explanation-validation endpoint.
{
 const s=add('AUROC tinggi menjawab pertanyaan yang berbeda','Masalah penelitian',
 'AUROC mengukur kemampuan prediksi membedakan kelompok. Reproduktibilitas peta menilai kesamaan distribusi attention antar pelatihan. Relevansi ranking menilai respons model ketika bagian input terpilih diganti. Kalibrasi menguji kecocokan probabilitas dengan frekuensi kejadian. Validitas fisiologis memerlukan rujukan yang sesuai. Skor klasifikasi tinggi tidak dengan sendirinya menjawab empat pertanyaan lain. Studi ini mengaudit ranking, bukan menguji korelasi universal antara AUROC dan kualitas penjelasan.',
 'Theissler et al. (2022); Turbé et al. (2023); Silva Filho et al. (2023); V9 Methods');
 card(s,'Keluaran','Apakah model membedakan PD dan kontrol?',0.65,1.87,3.85,2.2);
 card(s,'Peta','Apakah seed lain menyorot waktu yang sama?',4.75,1.87,3.85,2.2);
 card(s,'Ranking','Apakah bagian terpilih memengaruhi model?',8.85,1.87,3.85,2.2);
 R(s,0.65,4.55,12.05,1.23,C.light);
 T(s,'Kalibrasi dan validitas fisiologis adalah pertanyaan terpisah.',0.94,4.77,11.45,0.8,25,{bold:true});
 noteBox(s,'Fokus utama studi ini adalah relevansi ranking terhadap intervensi.');
}

// 03 — Show the whole output distribution's summary before a stable subset.
{
 const s=add('Perubahan keluaran terlihat sebelum subset dipilih','Discovery · tiga seed',
 'Tabel ini memakai bank discovery NewHandPD yang sama dengan E2. Setiap arsitektur mempunyai 792 pasangan tak terarah dari 264 rekaman dan tiga seed. Rerata perubahan probabilitas sekitar 0,20 untuk BiGRU dan 0,17 untuk BiMamba-2. Karena itu, seluruh keluaran tidak dapat disebut stabil. Selang berasal dari bootstrap subjek yang tersimpan. JSD adalah divergence basis dua. Besarnya JSD bukan ukuran penurunan kualitas. Contoh keluaran serupa baru ditampilkan sesudah gambaran seluruh pasangan ini.',
 'sources/discovery_e1_ci.csv; V9 Table e1_continuous');
 const rows=['gru','mamba2'].map(a=>{const r=m=>one(E.discovery_e1,r=>r.architecture===a&&r.metric===m,'e1');return [model(a),ci(r('abs_delta_prob')),ci(r('abs_delta_logit')),ci(r('js_divergence'))];});
 table(s,['Model','Rerata |Δp| [CI]','Rerata |Δz| [CI]','Rerata JSD [CI]'],rows,0.65,2.07,[2.0,3.2,3.3,3.55],1.02,21,0.67);
 T(s,'p = probabilitas PD · z = skor sebelum sigmoid · JSD = perbedaan distribusi attention',0.71,1.62,11.95,0.29,17,{color:C.muted});
 R(s,0.65,5.05,12.05,1.0,C.pale);T(s,'792 pasangan · 264 rekaman · 66 subjek per arsitektur',0.93,5.22,11.6,0.63,27,{bold:true});
 noteBox(s,'Rerata berbobot subjek. CI adalah selang kepercayaan nominal 95% pada bank discovery.');
}

// 04 — An actual, outcome-independent selected example, not a synthetic curve.
{
 const ex=E.examples.find(x=>x.architecture==='gru');
 const s=add('Pada rekaman yang sama, keluaran serupa dapat disertai peta berbeda','Contoh nyata · bank discovery',
 `Contoh ini berasal dari kontrol sehat H10, tugas spiral 1, segmen 0, fold 1. Dua checkpoint BiGRU memakai seed ${ex.seed_a} dan ${ex.seed_b}. Identitas lengkap dan hash ada pada sources/attention_examples.json. Kasus dipilih berdasarkan urutan leksikografis di antara pasangan yang memenuhi gate ketat dan matching kedua replacement. Pemilihan tidak diurutkan menurut JSD atau keuntungan ranking. Contoh ini memperlihatkan keberadaan fenomena. Ia bukan gambaran seluruh bank. Gate ketat hanya memilih 19 dari 792 pasangan BiGRU dan 52 dari 792 pasangan BiMamba-2. Kedua peta tetap memerlukan pengujian relevansi.`,
 'sources/attention_examples.json; sources/discovery_e1_sensitivity.csv');
 const labels=ex.alpha_a.map((_,i)=>String(i+1));
 chart(s,pptx.ChartType.line,[{name:`Seed ${ex.seed_a}`,labels,values:ex.alpha_a},{name:`Seed ${ex.seed_b}`,labels,values:ex.alpha_b}],
 {x:0.65,y:1.95,w:8.55,h:4.35,showLegend:true,chartColors:[C.own,C.source],lineSize:1.7,
  catAxisLabelFrequency:'25',showCatAxisTitle:true,catAxisTitle:'Indeks patch',catAxisTitleFontSize:14,
  valAxisTitle:'Bobot attention α',valAxisMinVal:0});
 T(s,`P(PD)\n${ex.probability_a.toFixed(6)}\n${ex.probability_b.toFixed(6)}`,9.5,2.07,2.95,1.55,25,{bold:true});
 T(s,`|Δlogit| = ${fmt(ex.abs_delta_logit)}\nJSD = ${fmt(ex.js_divergence)}`,9.5,3.96,2.95,1.0,21);
 noteBox(s,'Gate ketat discovery: BiGRU 19/792; BiMamba-2 52/792. Contoh bukan validasi fisiologis.');
}

// 05 — Scarcity through a concrete comparison, not global priority claims.
{
 const s=add('Bagaimana menilai perbedaan peta pada model yang sama?','Kesenjangan bukti',
 'Literatur terdekat sudah mengevaluasi beberapa dimensi penjelasan temporal. Bragança dan Souto menyediakan kerangka UTS-XAI. Šimić dan kolega menunjukkan pentingnya replacement dan ukuran region. Navakauskas dan Dumpis menguji regularisasi penjelasan temporal. Studi kita memusatkan satu perbandingan yang khusus. Ranking sumber diuji pada rekaman yang sama di model target yang parameternya tetap. Setiap mask mempunyai kontrol RMS dan geometri sendiri. Perbedaan ini menjelaskan nilai tambah desain tanpa klaim pertama di dunia. Ketiga studi adalah pembanding protokol, bukan leaderboard pada dataset handwriting yang sama.',
 'Bragança & Souto (2026), doi:10.1038/s41598-026-49467-2; Šimić et al. (2025), doi:10.1038/s41598-025-09538-2; Navakauskas & Dumpis (2026), doi:10.3390/electronics15102200');
 card(s,'UTS-XAI','Stabilitas dan mutu\npenjelasan temporal\n\nBragança & Souto, 2026',0.65,1.87,3.85,2.7);
 card(s,'Operator intervensi','Replacement dan\nukuran region\n\nŠimić et al., 2025',4.75,1.87,3.85,2.7);
 card(s,'Regularisasi peta','Penjelasan temporal\ndalam pelatihan\n\nNavakauskas & Dumpis, 2026',8.85,1.87,3.85,2.7);
 T(s,'Tambahan studi ini',0.75,5.0,3.4,0.5,24,{bold:true,color:C.transfer});
 T(s,'Ranking berbeda → rekaman yang sama → target tetap → kontrol terpadan',0.75,5.64,11.9,0.8,25,{bold:true});
}

// 06 — Task and subject denominators are not interchangeable.
{
 const s=add('Dua akuisisi menguji prosedur yang sama setelah retraining','Data penelitian',
 'NewHandPD menyumbang empat rekaman spiral per subjek. Terdapat 264 rekaman dari 66 subjek, dengan 31 PD dan 35 kontrol. Audit UCI hanya memakai static spiral test. SST memuat 77 rekaman dari 76 subjek, dengan 61 PD dan 15 kontrol. Satu subjek dapat memiliki lebih dari satu rekaman. Angka 77 subjek milik keseluruhan resource UCI tidak dipakai sebagai denominator SST. Dua akuisisi mempunyai kanal dan komposisi kelas berbeda. Model dilatih ulang terpisah. Ini bukan transfer bobot model UCI ke NewHandPD. HandPD, NewHandPD dan PaHaW tidak disamakan.',
 'Completed cohort manifests; UCI dataset doi:10.24432/C5Q01S; NewHandPD original dataset source; konteks analisis spiral: Toffoli et al. (2023)');
 card(s,'NewHandPD','264 rekaman · 66 subjek\n31 PD / 35 kontrol\n\nSmart pen · empat tugas spiral\nTurunan enam kanal sensor',0.65,1.93,5.85,3.45);
 card(s,'UCI 395 · SST','77 rekaman · 76 subjek\n61 PD / 15 kontrol\n\nTablet · template spiral statis\nDinamika koordinat dan tekanan',6.8,1.93,5.85,3.45);
 noteBox(s,'Resampling berbeda. Waktu UCI bersifat nominal; kedua dataset tidak dipool.');
}

// 07 — Separate scientific banks, not thirty seeds claimed in retrospect.
{
 const s=add('Bank 20 seed memperluas discovery, bukan menggantikan riwayatnya','Desain dan provenance',
 'Bank replikasi mempunyai dua arsitektur, dua dataset, dua puluh seed, dan lima fold per dataset. Jumlah identitas checkpoint adalah 400. Tiga puluh checkpoint NewHandPD discovery digunakan kembali. Sebanyak 370 checkpoint dilatih baru. Analisis primer NewHandPD membatasi target dan sumber pada seed 3 sampai 19. Analisis primer UCI menggunakan dua puluh seed baru. Bank sepuluh seed historis untuk AUROC terpisah dari keduanya. Protokol perluasan dikunci lokal pada 19 Oktober setelah hasil discovery diketahui. Ia tidak didaftarkan publik. Semua hasil tambahan tetap dilaporkan eksploratori.',
 'sources/newhandpd_manifest.json; sources/uci395_manifest.json; plans/attention_audit_replication20_20261019/design_lock.json');
 const cells=[['Historis','10 seed','AUROC dan probabilitas\nBank pendamping'],['Discovery','3 seed','30 checkpoint NewHandPD\nSumber temuan awal'],['Audit diperluas','20 seed','400 identitas checkpoint\nDua dataset × dua model']];
 cells.forEach((a,i)=>{card(s,a[0],a[2],0.65+i*4.1,2.75,3.85,2.6);T(s,a[1],0.88+i*4.1,1.9,3.35,0.65,36,{color:i===2?C.transfer:C.ink,bold:true});});
 noteBox(s,'Primer NewHandPD: seed 3–19. Primer UCI-SST: seed 0–19. Partisi subjek tetap.');
}

// 08 — The attention head is not a supervised physiological localizer.
{
 const s=add('Satu classifier menghasilkan probabilitas dan attention','Arsitektur dan praproses',
 'Model tetap dilatih untuk klasifikasi PD versus kontrol. Peta attention adalah bobot pooling yang diperoleh dari classifier tersebut. Model tidak dilatih dengan anotasi lokasi gejala. Input dipecah menjadi patch tujuh sampel. Encoder BiGRU atau BiMamba-2 membangun representasi kontekstual. Attention mengagregasi representasi menjadi vektor prediksi. Median, IQR dan batas clipping dihitung dari training fold. Semua rekaman satu subjek berada pada fold yang sama. Bobot attention yang besar belum menjamin kausalitas patch input karena representasi kontekstual dapat memuat informasi dari waktu lain.',
 'V9 Methods; Diaz et al. (2021); Qu et al. (2026); Kapoor & Narayanan (2023)');
 const boxes=[['Enam kanal','Praproses khusus dataset'],['Patch 7 sampel','Normalisasi train fold'],['BiGRU / BiMamba-2','Encoder kontekstual'],['Attention pooling','Peta α dan vektor v']];
 boxes.forEach((b,i)=>{card(s,b[0],b[1],0.65+i*3.1,2.1,2.8,2.1);if(i<3)A(s,3.5+i*3.1,3.15,3.72+i*3.1,3.15);});
 R(s,3.5,4.72,6.4,1.07,C.ink);T(s,'Logit z → probabilitas P(PD)',3.75,4.93,5.9,0.67,29,{color:C.white,bold:true});
 noteBox(s,'Parameter pelatihan: AdamW · lr 0,0003 · batch 8 · 35 epoch · checkpoint epoch terakhir.');
}

// 09 — Transfer selects input patches; it does not overwrite attention-head weights.
{
 const s=add('Hanya sumber ranking yang berubah saat target diuji','Ranking-swap pada target tetap',
 'Seed a adalah target. Seed b menyediakan ranking attention untuk rekaman held-out yang sama. Keduanya berasal dari arsitektur, dataset dan fold yang sama. Ranking a dan b menentukan patch input yang diganti. Semua input hasil intervensi dijalankan pada checkpoint target a yang sama. Normalizer target juga tetap. Kami tidak menyalin bobot attention ke kepala model target. Forward baru menghitung ulang representasi dan attention secara biasa. Jadwal lima sumber per target dipilih sebelum respons perluasan diperiksa. Pasangan terarah berbagi model dan subjek, sehingga bukan observasi independen.',
 'V9 fixed-target Methods; locked directed schedule; complete execution manifests');
 card(s,'Ranking target a','Peringkat dari checkpoint a',0.65,2.0,4.15,1.65);
 card(s,'Ranking sumber b','Arsitektur sama · seed berbeda',0.65,4.2,4.15,1.65);
 R(s,5.42,2.48,2.35,2.55,C.pale);T(s,'Pilih patch\nGanti input',5.63,2.94,1.95,1.32,25,{bold:true});
 A(s,4.83,2.84,5.37,3.15);A(s,4.83,5.0,5.37,4.4);
 R(s,8.48,2.48,4.12,2.55,C.ink);T(s,'Checkpoint target a\nTETAP',8.75,2.93,3.57,1.35,29,{color:C.white,bold:true});
 A(s,7.83,3.75,8.4,3.75);noteBox(s,'Yang dipertukarkan adalah urutan pemilihan patch input, bukan parameter model.');
}

// 10 — Controls and geometry are stated exactly.
{
 const s=add('Respons besar harus dibandingkan dengan intervensi yang sepadan','Kontrol RMS dan geometri',
 'Ranking attention dapat memilih bagian sinyal yang menghasilkan perubahan input besar. Kontrol acak kami mencocokkan RMS perubahan tersebut. Kontrol juga mempertahankan jumlah patch dan multiset panjang run kontigu. Setiap mask memiliki kontrol sendiri. Kontrol milik ranking sendiri dan milik ranking transfer tidak berarti kedua ranking mempunyai dosis langsung yang sama. Median memakai nilai nol dalam ruang normalisasi. Donor berasal dari training subject dengan tugas sama. Kuota dan search cap dikunci. Kegagalan mencari match dalam batas itu dicatat, tanpa memperlebar toleransi atau memilih kandidat berdasarkan respons target.',
 'V9 intervention Methods; Šimić et al. (2025); locked matching configuration');
 card(s,'Budget tetap','5% · 10% · 20% · 30%\n\nSemua enam kanal diganti\nTail tidak diganti',0.65,1.9,3.85,3.62);
 card(s,'Geometri tetap','Jumlah patch sama\nPanjang blok kontigu sama\n\nLokasi kontrol boleh berbeda',4.75,1.9,3.85,3.62);
 card(s,'Besar perubahan','RMS = root mean square\nBesar perubahan dicocokkan\n\nToleransi relatif 5%\nMaksimum 4096 kandidat',8.85,1.9,3.85,3.62);
 noteBox(s,'Median: 20 kontrol. Donor: 4 kontrol × 5 background. Pemilihan tidak memakai respons target.');
}

// 11 — Define the operational endpoint before results.
{
 const s=add('Keuntungan ranking dihitung terhadap kontrolnya sendiri','Definisi ukuran',
 'Logit adalah skor classifier sebelum sigmoid. Pada setiap budget, respons adalah besar perubahan logit dari kondisi tanpa masker. AOPC merangkum kurva itu pada budget aktual, lalu dinormalisasi dengan budget terakhir. T adalah AOPC ranking. R adalah AOPC kontrol terpadan. G sama dengan T dikurangi R. Nilai positif berarti respons ranking lebih besar daripada kontrolnya. D adalah G milik target dikurangi G dari ranking sumber pada target yang sama. D positif menguntungkan ranking sendiri. Karena respons absolut, gerak ke salah satu kelas dapat memperbesar skor. Ukuran ini mengukur pengaruh intervensi yang didefinisikan, bukan kebenaran fisiologis.',
 'V9 Equations for AOPC, G and D; Turbé et al. (2023); Šimić et al. (2025)');
 const equations=[['T','AOPC perubahan absolut logit','Respons terhadap input yang diganti'],['G = T − R','Keuntungan terhadap kontrol RMS','R adalah AOPC kontrol terpadan'],['D = G sendiri − G transfer','Keuntungan yang berubah saat ranking ditransfer','D positif menguntungkan ranking sendiri']];
 equations.forEach((q,i)=>{R(s,0.65,1.9+i*1.4,12.05,1.13,i===2?C.ink:C.pale);T(s,q[0],0.9,2.02+i*1.4,4.25,0.77,i===2?27:32,{bold:true,color:i===2?C.white:C.ink});T(s,q[1],5.35,1.99+i*1.4,7.05,0.48,23,{bold:true,color:i===2?C.white:C.ink});T(s,q[2],5.35,2.52+i*1.4,7.05,0.28,16,{color:i===2?'D2E6DD':C.muted});});
 noteBox(s,'G dan D memakai satuan logit-AOPC. Skala ini bukan ukuran universal antarmodel.');
}

// 12–13 — Paired advantages, nominal CIs, separate response scales.
for(const cohort of ['newhandpd','uci395']){
 const isN=cohort==='newhandpd', rows=primary(cohort);
 const s=add(isN?'NewHandPD menunjukkan transfer relevansi secara parsial':'UCI-SST mengulang urutan rata-rata yang sama',isN?'Hasil utama · NewHandPD':'Hasil utama · UCI-SST',
 `${isN?'NewHandPD menggunakan pasangan target dan sumber dari 17 seed baru.':'UCI-SST menggunakan seluruh 20 seed baru.'} Pada setiap baris, kedua ranking diuji pada support eligible yang sama. G sendiri dan G transfer mempunyai selang nominal positif. Selisih D juga positif di semua empat kondisi dataset ini. CI D dihitung dari kontras berpasangan, bukan dengan mengurangkan batas dua CI. Delapan uji D lintas kedua dataset menerima koreksi Holm. Semua p terkoreksi sekitar 0,001600 pada resolusi bootstrap yang tersedia. Bandingkan dua ranking dalam baris yang sama. Besar logit-AOPC bukan leaderboard arsitektur atau perbandingan universal antardataset.`,
 `sources/${cohort}_secondary_mean_contrasts.csv; sources/primary_eight_strata.csv`);
 T(s,'Keuntungan terhadap kontrol terpadan',0.7,1.66,8.0,0.33,19,{bold:true});
 bars(s,rows,[{name:'Ranking sendiri',get:r=>secondary(cohort,r,'G_own')},{name:'Ranking transfer',get:r=>secondary(cohort,r,'G_swap')}],0.5,2.04,8.45,4.3,isN?3.2:1.5,'G · logit-AOPC');
 T(s,'Selisih berpasangan D [CI]',9.25,1.95,3.42,0.6,21,{bold:true,color:C.own});
 rows.forEach((r,i)=>{T(s,condition(r),9.25,2.75+i*0.79,3.42,0.27,17,{bold:true});T(s,ci(r),9.25,3.1+i*0.79,3.42,0.38,19);});
 noteBox(s,'CI nominal 95% bootstrap subjek. Bandingkan ranking pada baris yang sama; skala dataset terpisah.');
}

// 14 — Coverage and stable counts have different axes and purposes.
{
 const s=add('Matching menentukan pasangan yang diwakili hasil','Cakupan hasil',
 'Cakupan tidak sama di setiap kondisi. NewHandPD BiMamba-2 median mempertahankan 8791 dari 20064 pasangan terencana, walaupun semua 66 subjek masih terwakili. UCI BiGRU median mempertahankan 5102 dari 7700 pasangan dan 67 dari 76 subjek. Panel ini memakai seluruh pasangan terencana sebagai denominator. Subset keluaran stabil berada pada panel terpisah karena jauh lebih kecil. Gate bukan analisis utama. Kegagalan search tidak membuktikan bahwa pencocokan matematis mustahil. Matching bisa mengubah populasi pasangan yang direpresentasikan; seluruh subjek yang hadir tidak menjamin seluruh pasangan tiap subjek tercakup.',
 'sources/newhandpd_matching_coverage.csv; sources/uci395_matching_coverage.csv');
 for(const [k,cohort] of ['newhandpd','uci395'].entries()){
  const rows=E.replication[cohort].matching_coverage;
  const labels=rows.map(condition).reverse();
  T(s,k?'UCI-SST':'NewHandPD',0.75+k*6.2,1.82,5.7,0.35,24,{bold:true});
  chart(s,pptx.ChartType.bar,[{name:'Matching lengkap',labels,values:[...rows].reverse().map(r=>100*Number(r.n_pairs_eligible)/Number(r.n_pairs_planned))}],
   {x:0.55+k*6.2,y:2.32,w:5.95,h:3.62,barDir:'bar',barGrouping:'clustered',chartColors:[C.transfer],valAxisMinVal:0,valAxisMaxVal:100,valAxisMajorUnit:25,
    valAxisTitle:'Pasangan eligible / terencana (%)',showValue:true,dataLabelFormatCode:'0.0',dataLabelPosition:'outEnd',dataLabelColor:C.text,dataLabelFontSize:13});
 }
 noteBox(s,'Denominator lengkap tersedia pada cadangan B. Grafik gate stabil memakai skala terpisah pada cadangan N.');
}

// 15 — Summaries of already estimated associations, not a new fit.
{
 const rows=['newhandpd','uci395'].flatMap(c=>E.replication[c].jsd_associations.filter(r=>r.endpoint==='abs_D'));
 const s=add('Perbedaan peta berkaitan dengan besar perbedaan relevansi','Hubungan kontinu · target tetap',
 'Analisis sekunder menghubungkan JSD dengan besar efek transfer absolut. Weighted Spearman positif pada delapan kondisi. Rentang koefisien NewHandPD sekitar 0,146 sampai 0,217. UCI sekitar 0,192 sampai 0,384. Ini asosiasi peringkat pada target yang tetap, bukan persamaan sebab-akibat. JSD menggunakan logaritma basis dua dan berada antara nol dan satu. Ia bukan akar divergence. Ranking yang identik memilih mask identik dan memberi D nol ketika kontrol bersama sama. Konstruksi ini dapat menyumbang hubungan. Oleh sebab itu, kami juga membandingkan ranking nyata dengan plasebo pada JSD serupa. Scatter bersumbu JSD penuh disimpan di cadangan.',
 'sources/newhandpd_jsd_associations.csv; sources/uci395_jsd_associations.csv; Nielsen (2021)');
 bars(s,rows,[{name:'Weighted Spearman',label:r=>(r.cohort==='newhandpd'?'NHPD':'UCI')+' · '+condition(r),get:r=>r}],0.4,1.85,8.2,4.22,0.55,'Koefisien ρ untuk JSD versus |D|',[C.transfer]);
 card(s,'Apa artinya?','Peta yang lebih berbeda\ncenderung disertai |D|\nyang lebih besar.\n\nIni asosiasi sekunder.',9.0,2.06,3.65,3.85);
 noteBox(s,'CI nominal. Target dan subjek digunakan berulang; placebos memeriksa penjelasan alternatif.');
}

// 16 — Retain all conditions, including zero-spanning intervals.
{
 const s=add('Ranking nyata tidak selalu melampaui peta plasebo','Kontrol pada JSD serupa',
 'Plasebo berasal dari circular shift atau permutasi peta target. Pemilihan kandidat memakai kedekatan JSD, bukan respons model. Setiap kandidat menerima kontrol sendiri. Matriks ini merangkum apakah CI nominal kontras ranking nyata minus plasebo berada di atas nol atau melintasi nol. Hasil berbeda antar arsitektur, replacement dan konstruksi plasebo. Selang yang melintasi nol bukan bukti kesetaraan. Cakupan juga dapat mengecil tajam. Contohnya, permutasi UCI BiGRU memenuhi pencocokan JSD pada 640 dari 7700 pasangan. Setelah kontrol lengkap, tersisa 406 pasangan donor dan 403 median. Angka dan CI penuh ada pada sumber serta cadangan.',
 'sources/newhandpd_placebo_contrasts.csv; sources/uci395_placebo_contrasts.csv');
 const xs=[0.65,3.46,5.72,7.98,10.24],ww=[2.81,2.26,2.26,2.26,2.46];
 const heads=['Dataset / model','Donor\nShift','Donor\nPermutasi','Median\nShift','Median\nPermutasi'];
 heads.forEach((v,i)=>{R(s,xs[i],1.98,ww[i],0.85,C.ink,C.white);T(s,v,xs[i]+0.1,2.07,ww[i]-0.2,0.66,18,{color:C.white,bold:true});});
 let j=0;for(const cohort of ['newhandpd','uci395'])for(const arch of ['gru','mamba2']){
  const y=2.83+j*0.7;R(s,xs[0],y,ww[0],0.7,j%2?C.white:C.light,C.white);T(s,(cohort==='newhandpd'?'NewHandPD':'UCI-SST')+'\n'+model(arch),xs[0]+0.1,y+0.08,ww[0]-0.2,0.54,18,{bold:true});
  let i=1;for(const m of ['train_donor','train_median'])for(const ref of ['circular_shift','permutation']){
   const r=one(E.replication[cohort].placebo_contrasts,r=>r.architecture===arch&&r.masker===m&&r.reference===ref,'placebo');
   const positive=Number(r.ci_low)>0;R(s,xs[i],y,ww[i],0.7,positive?'DDEDE4':'F0E9F2',C.white);T(s,positive?'CI di atas 0':'CI melintasi 0',xs[i]+0.08,y+0.1,ww[i]-0.16,0.48,17,{color:positive?C.own:C.source,bold:true});i++;
  }j++;
 }
 noteBox(s,'UCI BiGRU/permutasi: 406 donor dan 403 median complete; 44 dan 37 subjek. CI tanpa koreksi sekunder.');
}

// 17 — Dose restriction and loss of support are both shown.
{
 const cohort='uci395';const rows=primary(cohort).filter(r=>r.masker==='train_donor').map(r=>secondary(cohort,r,'D','dose_gap_at_most_5percent'));
 const s=add('Mencocokkan kontrol belum menyamakan dosis kedua ranking','Sensitivitas dosis',
 'Setiap mask sudah mempunyai kontrol yang terpadan. Namun, RMS mask ranking sendiri dapat berbeda dari RMS mask transfer. Sensitivitas tambahan membatasi gap langsung menjadi paling besar lima persen pada semua budget dan background. Kedua kondisi donor UCI mempunyai CI D yang melintasi nol pada support itu. BiGRU tinggal 139 pasangan dari dua puluh subjek. BiMamba-2 tinggal 71 pasangan dari sebelas subjek. Enam kondisi lain mempunyai CI nominal positif. Perubahan support harus dibaca bersama perubahan interval. Pembatasan dosis ini tetap diagnostik dan tidak mengidentifikasi efek kausal.',
 'sources/uci395_secondary_mean_contrasts.csv; sources/newhandpd_secondary_mean_contrasts.csv');
 table(s,['UCI-SST · donor','D setelah gap dosis ≤5% [CI]','Pasangan','Subjek'],rows.map(r=>[model(r.architecture),ci(r),r.n_pairs,r.n_subjects]),0.65,2.1,[3.2,4.65,2.1,2.1],1.05,22,0.66);
 R(s,0.65,5.0,12.05,1.15,C.pale);T(s,'Enam kondisi lain mempertahankan CI nominal positif.',0.94,5.19,11.45,0.72,26,{bold:true});
 noteBox(s,'Support yang lebih kecil membatasi perbandingan langsung dengan estimasi utama.');
}

// 18 — Additional seeds are not extra participants.
{
 const s=add('Rerata yang konsisten tetap menyisakan variasi target','Variasi training dan inferensi',
 'Leave-one-seed-out mengeluarkan seed dari peran target maupun sumber. Semua rerata D dari pemeriksaan tersimpan tetap positif. Beberapa rerata target BiGRU negatif. Karena itu, urutan rata-rata tidak berlaku untuk setiap checkpoint. Bootstrap mengulang subjek beserta rekaman dan pasangan di dalamnya. Ia bersyarat pada bank, split dan jadwal. Banyak pasangan yang berbagi model tidak dianggap sebagai banyak pelatihan independen. Dua puluh seed membantu memeriksa variasi training, tetapi tidak menambah peserta. Besar efek discovery juga tidak dianggap identik dengan hasil seed baru.',
 'sources/*_target_seed_effects.csv; sources/*_leave_one_seed_out.csv; sources/newhandpd_new_minus_discovery_subject_contrast.csv; Loy & Korobova (2022); Lakens (2022)');
 card(s,'Pengaruh seed','Semua mean D tetap positif\nsaat satu seed dikeluarkan.\n\nSebagian target BiGRU\nmempunyai mean D negatif.',0.65,1.93,5.85,3.6);
 card(s,'Unit ketidakpastian','Bootstrap berdasarkan subjek.\nPasangan nested dipertahankan.\n\nCI bersyarat pada model,\nsplit, jadwal dan matching.',6.8,1.93,5.85,3.6);
 noteBox(s,'400 checkpoint tidak berarti 400 kohort atau ratusan subjek baru.');
}

// 19 — Benefits follow from the evidence rather than a claimed clinical use.
{
 const s=add('Audit memberi tiga pemeriksaan untuk membandingkan penjelasan','Kontribusi dan manfaat',
 'Kontribusi pertama adalah pengujian ranking pada target yang parameternya tetap. Ia membantu membedakan perubahan fungsi model dari perubahan lokasi input yang dipilih. Kontribusi kedua adalah kontrol terhadap besar dan geometri intervensi. Ia membantu menilai apakah keuntungan attention hanya mengikuti perubahan input yang lebih kuat. Kontribusi ketiga adalah replikasi seed dan akuisisi dengan pelaporan coverage. Ia menunjukkan pola yang bertahan dan kondisi yang lebih terbatas. Manfaat praktisnya adalah debugging dan evaluasi penjelasan. Hasil tersebut bukan validasi untuk diagnosis klinis otomatis.',
 'V9 Results and Discussion; paired transfer, RMS controls, two-dataset replication');
 const list=[['Uji pada target tetap','Bandingkan relevansi ranking yang berbeda.'],['Kontrol intervensinya','Pisahkan pengaruh ranking dari besar perubahan input.'],['Baca efek bersama coverage','Kenali kasus yang diwakili oleh hasil.']];
 list.forEach((q,i)=>{R(s,0.65,1.95+i*1.4,0.82,0.82,C.own);T(s,String(i+1),0.8,2.08+i*1.4,0.53,0.52,28,{color:C.white,bold:true});T(s,q[0],1.83,1.96+i*1.4,10.55,0.48,27,{bold:true});T(s,q[1],1.83,2.59+i*1.4,10.55,0.52,23);});
}

// 20 — Put the general scientific boundary in one place.
{
 const s=add('Bukti menjangkau dua akuisisi, dengan cakupan yang terdefinisi','Cakupan dan tindak lanjut',
 'Studi ini memeriksa dua akuisisi spiral dan dua keluarga encoder setelah retraining terpisah. Tiap dataset masih mempunyai jumlah subjek terbatas. Split sama dipertahankan antar seed. Hasil primer mencakup pasangan dengan kontrol lengkap. Placebo dan dosis menunjukkan kondisi yang memerlukan pembacaan lebih khusus. Replikasi dengan split baru atau kohort lebih besar dapat memperluas cakupan. Baseline atribusi lain dapat menguji apakah pola juga muncul di luar attention. Anotasi temporal independen diperlukan untuk klaim fisiologis. E3 robustness input menjawab pertanyaan lain dan tidak dijalankan untuk presentasi ini.',
 'V9 Discussion; matching/eligibility diagnostics; Kapoor & Narayanan (2023); Lakens (2022)');
 card(s,'Bukti tersedia','Dua dataset spiral\nDua arsitektur\nBank 20 seed\nTarget tetap dan kontrol RMS',0.65,1.93,5.85,3.65);
 card(s,'Perluasan berikutnya','Partisi dan peserta baru\nMetode atribusi lain\nAnotasi temporal independen\nRobustness terhadap input',6.8,1.93,5.85,3.65);
 noteBox(s,'Analisis tambahan eksploratori. Holm mencakup 8 uji primer; sensitivitas memakai CI nominal.');
}

// 21 — A strong, bounded answer to the title.
{
 const s=add('Relevansinya berpindah sebagian','Kesimpulan',
 'Jawaban ringkasnya adalah transfer parsial. Pada pasangan dengan kontrol lengkap, ranking dari seed lain tetap mempunyai keuntungan rata-rata terhadap kontrol. Ranking milik target memberikan keuntungan lebih besar. Besar perbedaan peta berasosiasi dengan besar perbedaan keuntungan ranking. Placebo memperlihatkan variasi antar kondisi. Karena itu, penjelasan yang berbeda tidak langsung dinilai buruk hanya dari kemiripan peta. Relevansinya perlu diuji pada model dan intervensi yang jelas. AUROC tetap penting untuk klasifikasi, tetapi tidak menggantikan pengujian penjelasan.',
 'V9 primary and secondary results; conclusion',false,true);
 T(s,'Relevansinya\nberpindah sebagian.',0.72,0.9,11.9,1.55,44,{color:C.white,bold:true});
 const takeaways=['Ranking dari seed lain tetap memiliki keuntungan rata-rata.', 'Ranking milik target memberi keuntungan lebih besar.', 'Perbedaan peta perlu dinilai melalui pengaruhnya pada model.'];
 takeaways.forEach((q,i)=>{T(s,String(i+1),0.79,3.0+i*0.9,0.55,0.55,27,{color:'A2D5C0',bold:true});T(s,q,1.62,3.0+i*0.9,10.8,0.6,26,{color:C.white});});
 T(s,'Temuan bersyarat pada support matching dan intervensi yang diuji.',0.79,6.36,11.7,0.5,18,{color:'D2E6DD'});
}

// BACKUP A — Exact primary contrasts and denominators.
{
 const s=add('A · Delapan kontras primer pada support lengkap','Cadangan · angka utama',
 'Tabel ini memberikan angka per kondisi. CI adalah interval percentile nominal. Nilai p Holm memakai satu keluarga delapan uji primer. Semua D positif pada estimasi tersimpan. Itu bukan uji superioritas satu encoder atas encoder lain. Skala respons berbeda antar model dan dataset. Denominator subjek mengikuti matching masing-masing kondisi.',
 'sources/primary_eight_strata.csv; konteks koreksi multiplicity pada data berkelompok: Watson et al. (2023)',true);
 table(s,['Dataset','Model / masker','Mean D [CI]','Pasangan','Subjek'],E.primary.map(r=>[r.cohort==='newhandpd'?'NewHandPD':'UCI-SST',condition(r),ci(r),r.n_pairs,r.n_subjects]),0.65,1.83,[1.75,3.05,4.15,1.7,1.4],0.5,17,0.57);
 noteBox(s,'p Holm = 0,001600 pada delapan kontras. CI tetap nominal 95%, bukan interval simultan.');
}

// BACKUP B — Distinct matching and stable denominators, with full numbers.
{
 const s=add('B · Cakupan matching dan subset keluaran stabil','Cadangan · denominator',
 'Gate stabil mensyaratkan kelas sama, selisih probabilitas paling besar 0,02 dan selisih logit paling besar 0,1. Kolom gate memakai pasangan terencana sebagai denominator. Kolom setelah matching menunjukkan irisan dua kriteria. Angka ini berasal dari bank replikasi dan tidak ditukar dengan persentase discovery tiga seed. NewHandPD BiMamba-2 median mempertahankan semua subjek, tetapi tidak semua pasangan di dalam subjek. UCI BiGRU median kehilangan beberapa subjek.',
 'sources/newhandpd_matching_coverage.csv; sources/uci395_matching_coverage.csv',true);
 const rows=['newhandpd','uci395'].flatMap(c=>E.replication[c].matching_coverage.map(r=>[(c==='newhandpd'?'NHPD':'UCI')+' · '+condition(r),r.n_pairs_eligible+'/'+r.n_pairs_planned,r.n_subjects_eligible+'/'+r.n_subjects_planned,r.n_stable_planned+'/'+r.n_pairs_planned,r.n_stable_eligible]));
 table(s,['Kondisi','Match lengkap','Subjek','Gate sebelum match','Gate + match'],rows,0.65,1.83,[3.52,2.42,1.22,2.75,2.14],0.5,17,0.62);
 noteBox(s,'Analisis kontinu menggunakan seluruh pasangan eligible. Gate hanya sensitivitas.');
}

// BACKUP C — The actual training recipe.
{
 const s=add('C · Resep training dan pasangan yang dikunci','Cadangan · reproduktibilitas',
 'Semua konfigurasi berasal dari design lock dan implementasi training yang dipatok. Checkpoint dipilih pada epoch terakhir. Tidak ada early stopping atau tuning baru pada replikasi ini. Lima fold subjek dengan split seed 42 dipertahankan antar seed. Jadwal sumber memakai ring yang dikunci. Setiap target mempunyai lima sumber pada bank dua puluh seed. Subgraph primer NewHandPD hanya memakai seed baru di kedua peran. Sumber yang tersedia dirata-rata di dalam target sebelum target dirata-rata.',
 'plans/attention_audit_replication20_20261019/design_lock.json; V9 configuration table',true);
 table(s,['Komponen','Konfigurasi'],[['Encoder','BiGRU / BiMamba-2'],['Representasi','d_model 128 · 3 layer · dropout 0,30'],['Input','6 kanal · patch 7 sampel'],['Training','AdamW · lr 0,0003 · batch 8 · 35 epoch'],['Pemilihan model','Epoch terakhir · normalizer hanya training fold'],['Validasi','5 fold subjek · split seed 42'],['Jadwal transfer','5 sumber/target · scheduling seed 2101'],['Inferensi','5000 draw subjek · seed 2102 · Holm 8 kontras']],0.65,1.85,[3.3,8.75],0.5,20,0.58);
}

// BACKUP D — Do not retroactively turn the changing-model bridge into fixed-target swap.
{
 const s=add('D · Bridge discovery mengestimasi hubungan yang berbeda','Cadangan · fungsi model berubah',
 'Bridge discovery membandingkan perubahan peta dengan perubahan keuntungan saat fungsi model juga berganti seed. Weighted Spearman mentah adalah analisis primer bridge. Normalisasi menjadi sensitivitas. Pearson tersentrik mengestimasi asosiasi linear di dalam subjek. Seluruh CI Spearman normalized mencakup nol. Pearson normalized BiGRU positif pada kedua replacement. Karena estimator dan objek evaluasi berbeda, hasil tersebut tidak bertentangan dengan asosiasi JSD versus efek transfer pada target tetap. Signifikan di satu kondisi dan tidak di kondisi lain bukan uji perbedaan antar replacement.',
 'sources/discovery_bridge_associations.csv; V9 bridge tables',true);
 const get=(arch,m,y,est,support)=>one(E.bridge,r=>r.architecture===arch&&r.masker===m&&r.level_id==='aopc_all_budgets'&&r.scenario==='all_pairs'&&r.x==='js_divergence'&&Number(r.denominator_floor)===1e-8&&r.y===y&&r.estimator===est&&r.support===support,'bridge display');
 const rows=[];for(const a of ['gru','mamba2'])for(const m of ['train_donor','train_median']){
  const raw=get(a,m,'abs_delta_gain_abs_delta_logit','subject_balanced_weighted_spearman','pairwise_complete_raw');
  const norm=get(a,m,'abs_delta_relative_gain','subject_balanced_weighted_spearman','same_normalization_valid_support');
  const pear=E.bridge.filter(r=>r.architecture===a&&r.masker===m&&r.level_id==='aopc_all_budgets'&&r.scenario==='all_pairs'&&r.x==='js_divergence'&&Number(r.denominator_floor)===1e-8&&r.y==='abs_delta_relative_gain'&&r.estimator.includes('pearson'));
  if(pear.length!==1)throw Error('Pearson selector '+pear.length);
  rows.push([model(a)+' / '+masker(m),ci(raw,4),ci(norm,4),ci(pear[0],4)]);
 }
 table(s,['Model / masker','Spearman raw','Spearman normalized','Pearson centered normalized'],rows,0.65,1.97,[2.55,3.0,3.05,3.45],0.89,17,0.86);
 noteBox(s,'Bank discovery · tiga seed. Pearson centered bukan pengganti estimand Spearman primer.');
}

// BACKUP E — Preserve an unfavourable comparator rather than hiding it.
{
 const rows=E.amplitude.filter(r=>r.scope==='full_baseline'&&r.metric==='paired_magnitude_abs_delta_logit');
 const s=add('E · Magnitude ranking juga dapat mengungguli attention','Cadangan · pembanding discovery',
 'Tabel ini berasal dari kontrol amplitude-aware pada discovery tiga seed. Nilai adalah respons attention dikurangi magnitude ranking untuk AOPC perubahan absolut logit. Negatif berarti magnitude lebih besar. Pada support baseline lengkap, magnitude mengungguli attention untuk kedua replacement BiMamba-2. Pada irisan yang memenuhi kontrol RMS lengkap E2, donor tetap negatif dan CI median melintasi nol. Setiap support menggunakan hasil bootstrap yang sudah tersimpan. Irisan tidak berarti magnitude menerima pencocokan dosis dan geometri. Pembanding magnitude belum menyamakan keduanya langsung dengan attention. Hasil ini bukan peringkat universal mutu penjelasan.',
 'sources/discovery_amplitude_ci.csv; V9 E2 controls',true);
 if(rows.length!==4)throw Error('magnitude selector '+rows.length);
 const comparison=rows.map(r=>{const common=one(E.amplitude,q=>q.architecture===r.architecture&&q.masker===r.masker&&q.scope==='matched_common_support'&&q.metric===r.metric,'magnitude common');return [condition(r),ci(r)+'\n'+r.n_subjects_valid+' subjek',ci(common)+'\n'+common.n_subjects_valid+' subjek'];});
 table(s,['Model / masker','Baseline lengkap\nAttention − magnitude [CI]','Irisan lengkap E2\nAttention − magnitude [CI]'],comparison,0.65,1.95,[3.15,4.45,4.45],0.84,18,0.78);
 noteBox(s,'Selisih respons memakai logit-AOPC. Magnitude belum menerima matching dosis/geometri pada kedua kolom.');
}

// BACKUP F — Historical classification consistency is neither a transfer claim nor an accuracy metric.
{
 const s=add('F · BiMamba-2 lebih konsisten secara deskriptif setelah retraining','Cadangan · AUROC historis',
 'Ini bank historis sepuluh seed, bukan checkpoint audit dua puluh seed. Probabilitas rekaman dirata-rata dalam subjek. AUROC dihitung per seed lalu dirata-rata antar seed. BiMamba-2 lebih rendah pada UCI dan lebih tinggi pada NewHandPD. Kontras penurunan Mamba minus GRU adalah sekitar minus 0,0514, dengan interval minus 0,10678 sampai 0,00074. Interval ini masih mencakup nol. Temuan mendukung konsistensi arsitektur secara deskriptif setelah retraining terpisah. Ia tidak menetapkan superioritas retensi, equivalence atau keberhasilan model beku pada kohort baru.',
 'sources/historical_auroc.csv; sources/historical_auroc_contrast.csv',true);
 table(s,['Dataset','Model','AUROC [CI]'],E.historical_auc.map(r=>[r.cohort==='uci395'?'UCI 395 historis':'NewHandPD',model(r.architecture),ci({...r,estimate:r.mean_auroc,_estimateField:'mean_auroc'},4)]),0.65,1.96,[3.4,3.05,5.6],0.73,23,0.65);
 const c=E.historical_auc_contrast[0];T(s,`Kontras penurunan = ${fmt(c.contrast,4)} [${fmt(c.ci_low,5)}; ${fmt(c.ci_high,5)}]`,0.77,5.83,11.8,0.64,25,{bold:true});
}

// BACKUP G — Brier is probability quality; ECE is a different endpoint.
{
 const rows=E.calibration.filter(r=>r.bank==='current_E1_E2'&&(r.metric==='brier'||r.metric==='ece_equal_width_10'));
 const s=add('G · Discovery Brier menguntungkan BiMamba-2 pada kualitas probabilitas','Cadangan · probabilitas OOF',
 'Analisis ini memakai bank discovery tiga seed dengan 66 subjek. Ia tidak mengevaluasi kalibrasi bank replikasi dua puluh seed. Brier menghitung kualitas probabilitas secara keseluruhan. ECE utama memakai sepuluh bin sama lebar dari probabilitas PD, bukan confidence kelas maksimum. Metrik dihitung per seed setelah agregasi probabilitas dalam subjek. Kontras Brier berpasangan menguntungkan BiMamba-2 dan CI tidak melintasi nol. Kontras ECE utama masih melintasi nol. Sensitivitas lima bin dan sepuluh quantile dipertahankan. Tidak ada fitting calibrator pada data evaluasi. Ensemble tidak dijalankan.',
 'sources/calibration_scores.csv; sources/calibration_contrasts.csv; Silva Filho et al. (2023); Dimitriadis et al. (2021)',true);
 table(s,['Model','Ukuran','Estimasi [CI]'],rows.map(r=>[model(r.architecture),r.metric==='brier'?'Brier':'ECE · 10 bin',ci(r,4)]),0.65,1.91,[3.0,3.5,5.55],0.69,22,0.62);
 const b=one(E.calibration_contrasts,r=>r.bank==='current_E1_E2'&&r.metric==='brier','Brier');
 T(s,`Brier BiMamba-2 − BiGRU = ${ci(b,4)}`,0.77,5.6,11.8,0.75,25,{bold:true});
 noteBox(s,'Brier bukan ukuran kalibrasi murni. Perhitungan ensemble tetap terpisah dan tidak dijalankan.');
}

// BACKUP H — Historical context is not mixed into the current model bank.
{
 const s=add('H · Kontrol historis membatasi interpretasi diagnostik','Cadangan · usia, fitur dan marker',
 'Analisis historis NewHandPD menunjukkan selisih usia sekitar 14,7 tahun dan age-only AUROC 0,8097. Restriksi rentang usia bukan matching pasangan yang seimbang. Baseline random forest historis menghasilkan UCI AUROC sempurna pada kohort kecil, tetapi memakai fitur agregat dan lima seed. Imputasi median global mendahului fold. Dampak aktual imputasi tersebut belum diisolasi. Marker sensor bukan ground truth fisiologis independen. Agreement Shapley custom dan ensemble juga bukan pemulihan fidelity yang tervalidasi. Konteks ini tidak disajikan seolah berasal dari checkpoint replikasi terbaru. Klaim utama presentasi tetap perilaku model terhadap intervensi.',
 'V9 Appendix Historical controls; results/p5_*; results/a_baseline_*; prior stored analyses',true);
 card(s,'Usia','Selisih historis ≈14,7 tahun\nAge-only AUROC 0,8097\n\nInterpretasi penyakit-spesifik\nmemerlukan kontrol tambahan.',0.65,1.94,3.85,3.97);
 card(s,'Baseline klasik','Random forest UCI 1,0000\nBank fitur agregat · 5 seed\n\nImputasi global mendahului fold.\nBukan leaderboard setara.',4.75,1.94,3.85,3.97);
 card(s,'Marker dan ensemble','Marker sensor adalah proksi.\nShapley custom adalah referensi.\n\nAgreement terhadap referensi ≠\nvalidasi fisiologis.',8.85,1.94,3.85,3.97);
}

// BACKUPS I/J — Full empirical axes, not an invented scatter.
for(const [cohort,label,file] of [['newhandpd','I','jsd_newhandpd_v7.png'],['uci395','J','jsd_uci_v7.png']]){
 const s=add(`${label} · Sebaran JSD dan |D| pada ${cohort==='newhandpd'?'NewHandPD':'UCI-SST'}`,'Cadangan · visual empiris lengkap',
 'Gambar ini adalah visual hasil tersimpan dari V9. Tidak ada fit atau agregasi baru untuk presentasi. Sumbu divergence ditampilkan penuh dari nol sampai satu. Banyak titik terkonsentrasi pada JSD tinggi. Titik berbagi subjek dan model, sehingga jumlah titik bukan ukuran sampel independen. Warna menandai perubahan probabilitas source versus target sebelum intervensi. Asosiasi memakai weighted Spearman dan bootstrap subjek yang tersimpan.',
 `V9 figures/${file}; stored paired results`,true);
 s.addImage({path:path.join(ROOT,'assets',file),...containPng(path.join(ROOT,'assets',file),0.68,1.76,11.95,4.98),altText:'Full-range JSD scatter from the saved V9 results'});
}

// BACKUP K — Where the new and historical claims differ.
{
 const s=add('K · Apa yang berubah dari presentasi September?','Cadangan · ketepatan klaim',
 'Materi September masih berupa rencana dan mencantumkan tiga encoder. Presentasi sekarang menyampaikan hasil selesai dengan dua arsitektur utama. Skema lama yang tampak seperti UCI train ke NewHandPD test diperbaiki menjadi retraining terpisah. Judul map breaks, rasio 4 sampai 95 kali, fidelity ceiling dan independensi marker tidak dibawa sebagai klaim sekarang. Status publik preregistration tetap false. Shapley tidak dipakai sebagai ground truth fisiologis. Bukti ketahanan BiMamba-2 tetap dipertahankan sebagai konsistensi deskriptif pada bank historis. Pemetaan lengkap tersedia dalam notes/PETA_PRESENTASI_LAMA_KE_BARU.md.',
 'Old presentation; V9; complete manifests; notes/CLAIM_AUDIT.md',true);
 table(s,['Materi September','Presentasi terbaru'],[['Rencana eksperimen dan tabel kosong','Hasil selesai dengan sumber angka'],['Tiga encoder sebagai pusat cerita','BiGRU dan BiMamba-2 · 20 seed'],['Kesan transfer antarkohort','Retraining terpisah per dataset'],['Shapley/marker sebagai ground truth','Referensi atribusi dan proksi sensor'],['Map breaks / fidelity ceiling','Transfer ranking dan kontrol RMS'],['Praregistrasi publik tersirat','Spesifikasi lokal; eksploratori']],0.65,1.95,[5.7,6.35],0.64,20,0.62);
}

// BACKUP L/M — Journal references only; acquisition attribution is separate.
for(const [page,keys] of [['L',['theissler2022','diaz2021','toffoli2023','braganca2026','simic2025','navakauskas2026','mambasurvey2026']],['M',['turbe2023','nielsen2021','kapoor2023','silva2023','dimitriadis2021','loy2022','watson2023','lakens2022']]]){
 const s=add(`${page} · Rujukan jurnal yang digunakan`,'Cadangan · sumber ilmiah',
 'Daftar ini berisi artikel jurnal dari tahun 2021 ke atas yang dipakai untuk konteks dan metode presentasi. Ketiga pembanding audit adalah pembanding protokol. Mereka bukan tiga studi dengan dataset, input dan subject split identik. Originalitas metode tidak dialihkan kepada penulis tinjauan. Atribusi akuisisi dataset dicatat terpisah di README dan sumber data karena jenis publikasinya berbeda.',
 'sources/references.json; V9 verified bibliography',true);
 const rowH=page==='L'?0.67:0.58;
 keys.forEach((key,i)=>{const r=E.references.find(r=>r.key===key);const surname=r.author.split(' and ')[0].split(',')[0];
  T(s,`${surname}${r.author.includes(' and ')?' et al.':''} (${r.year}) · ${r.journal}`,0.74,1.87+i*rowH,11.9,0.28,16,{bold:true});
  T(s,r.title,0.74,2.16+i*rowH,11.9,0.26,13,{color:C.muted});
 });
}

// BACKUP N — A smaller, explicitly separate axis keeps the stable subset visible.
{
 const s=add('N · Gate keluaran stabil memakai skala tersendiri','Cadangan · subset sensitivitas',
 'Kedua panel menggunakan maksimum dua puluh persen. Grafik matching utama memakai seratus persen. Denominator kedua seri di sini tetap seluruh pasangan terencana. Karena itu, seri setelah matching menunjukkan berapa banyak pasangan yang memenuhi gate sekaligus kontrol lengkap. Gate mensyaratkan kelas sama, perubahan probabilitas paling besar 0,02, dan perubahan logit paling besar 0,1. Subset ini adalah sensitivitas, bukan penjelasan bahwa semua keluaran stabil. Angka discovery 19 atau 52 dari 792 tidak digunakan sebagai persentase bank replikasi.',
 'sources/newhandpd_matching_coverage.csv; sources/uci395_matching_coverage.csv',true);
 for(const [i,cohort] of ['newhandpd','uci395'].entries()){
  const rows=[...E.replication[cohort].matching_coverage].reverse(),labels=rows.map(condition);
  T(s,i?'UCI-SST':'NewHandPD',0.75+i*6.2,1.8,5.7,0.35,24,{bold:true});
  chart(s,pptx.ChartType.bar,[
   {name:'Gate sebelum matching',labels,values:rows.map(r=>100*Number(r.n_stable_planned)/Number(r.n_pairs_planned))},
   {name:'Gate + matching',labels,values:rows.map(r=>100*Number(r.n_stable_eligible)/Number(r.n_pairs_planned))}],
   {x:0.55+i*6.2,y:2.3,w:5.95,h:3.9,barDir:'bar',barGrouping:'clustered',chartColors:['87949A','33434A'],
    showLegend:true,legendFontSize:12,valAxisMinVal:0,valAxisMaxVal:20,valAxisMajorUnit:5,
    valAxisTitle:'Pasangan / seluruh pasangan terencana (%)',showValue:true,dataLabelFormatCode:'0.0',
    dataLabelPosition:'outEnd',dataLabelColor:C.text,dataLabelFontSize:11});
 }
 noteBox(s,'Skala 0–20% menampilkan subset kecil; grafik matching utama memakai skala 0–100%.');
}

// Finish all footers after the final slide count is known.
slides.forEach((meta,i)=>{
 const s=pptx._slides[i];const color=meta.dark?'C8DDD2':C.muted;
 const citations={2:'Theissler et al., 2022 · Turbé et al., 2023 · Silva Filho et al., 2023',
  3:'NewHandPD discovery · 3 seed · 66 subjek · CI nominal 95%',
  4:'Contoh H10 · fold 1 · seed 0/1 · pemilihan leksikografis tanpa pengurutan berdasarkan JSD',
  5:'Bragança & Souto, 2026 · Šimić et al., 2025 · Navakauskas & Dumpis, 2026',
  6:'UCI doi:10.24432/C5Q01S · NewHandPD: sumber resmi UNESP · konteks spiral: Toffoli et al., 2023',
  7:'Spesifikasi lokal 19 Oktober 2026 · tidak didaftarkan publik',
  8:'Diaz et al., 2021 · Qu et al., 2026 · Kapoor & Narayanan, 2023',
  10:'Šimić et al., 2025 · konfigurasi kontrol yang dikunci sebelum perluasan',
  11:'Definisi operasional studi ini · konteks evaluasi: Turbé et al., 2023; Šimić et al., 2025',
  12:'Run NewHandPD selesai 23 Oktober 2026 UTC · primer seed 3–19',
  13:'Run UCI-SST selesai 25 Oktober 2026 UTC · primer seed 0–19',
  15:'Weighted Spearman sekunder · JSD basis 2 · Nielsen, 2021',
  18:'Loy & Korobova, 2022 · Lakens, 2022 · diagnostik seed dipisahkan dari CI subjek',
  20:'Kapoor & Narayanan, 2023 · Lakens, 2022',
  22:'Kontras primer tersimpan · konteks multiplicity: Watson et al., 2023',
  28:'Bank discovery · 3 seed · Silva Filho et al., 2023; Dimitriadis et al., 2021'};
 T(s,citations[i+1]||(meta.backup?'CADANGAN · sumber lengkap di catatan pembicara':'AUDIT ATTENTION TEMPORAL · 16 NOV 2026'),0.62,7.02,10.95,0.22,10,{color});
 T(s,`${i+1} / ${slides.length}`,11.9,6.99,0.8,0.29,12,{color,align:'right'});
});
fs.writeFileSync(path.join(ROOT,'review/CHART_BINDINGS.json'),JSON.stringify(chartSpecs,null,2)+'\n');
fs.writeFileSync(path.join(ROOT,'review/DISPLAY_CI_BINDINGS.json'),JSON.stringify(displayBindings,null,2)+'\n');
fs.writeFileSync(path.join(ROOT,'review/SHAPE_LAYOUT.json'),JSON.stringify(shapeLog,null,2)+'\n');
fs.writeFileSync(path.join(ROOT,'SLIDE_PLAN.json'),JSON.stringify({title:pptx.title,main_slides:21,total_slides:slides.length,slides},null,2)+'\n');
const markdown=['# '+pptx.title,'','Presentasi 16 November 2026. Bahasa Indonesia. Hasil tersimpan hingga 25 Oktober 2026.',''];
slides.forEach(s=>markdown.push(`## ${String(s.number).padStart(2,'0')} · ${s.title}`,`*${s.section}${s.backup?' — cadangan':''}*`,'',s.notes,'',`Sumber: ${s.source}`,'','---',''));
fs.writeFileSync(path.join(ROOT,'CATATAN_PEMBICARA.md'),markdown.join('\n'));
pptx.writeFile({fileName:path.join(ROOT,'review/presentasi_raw.pptx')}).then(()=>console.log('Built presentation only:',slides.length,'slides;',chartNumber,'native charts.'));
