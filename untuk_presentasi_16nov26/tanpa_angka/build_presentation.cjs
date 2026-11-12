/* Qualitative presentation only. Do not run research code or estimate results. */
const fs = require('fs');
const path = require('path');
const pptxgen = require('../node_modules/pptxgenjs');
const ROOT = __dirname;
const E = JSON.parse(fs.readFileSync(path.join(ROOT, '../sources/evidence.json'), 'utf8'));
const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Agriby Diandra Chaniago';
pptx.company = 'Universitas Harapan Bangsa';
pptx.title = 'Beda Peta, Masih Relevan? — Versi Tanpa Angka Hasil';
pptx.subject = 'Audit transfer ranking attention pada tulisan tangan Parkinsonian';
pptx.lang = 'id-ID';
pptx.theme = {headFontFace:'Arial', bodyFontFace:'Arial', lang:'id-ID'};
const C = {ink:'18342F', text:'22362F', muted:'52645F', pale:'EDF4F1', white:'FFFFFF',
  own:'173D35', transfer:'248D77', donor:'117A74', median:'964F79', source:'6F4D86',
  warn:'8C4F2B', warnPale:'F7EEE8', grid:'D7E2DD', light:'F6F8F7'};
const slides=[], shapes=[], charts=[];
function text(s, value, x,y,w,h,size=24, opts={}) {
  shapes.push({slide:s._n, kind:'text', text:value,x,y,w,h,size});
  s.addText(value,{x,y,w,h,fontFace:'Arial',fontSize:size,color:C.text,margin:0,
    breakLine:false,valign:'mid',paraSpaceAfterPt:0,...opts});
}
function box(s,x,y,w,h,fill=C.pale) {
  shapes.push({slide:s._n,kind:'shape',x,y,w,h});
  s.addShape(pptx.ShapeType.roundRect,{x,y,w,h,rectRadius:0.12,
    fill:{color:fill},line:{color:fill,width:0.5}});
}
function arrow(s,x,y,w,h=0,color=C.muted) {
  s.addShape(pptx.ShapeType.line,{x,y,w,h,line:{color,width:1.8,endArrowType:'triangle'}});
}
function add(title,section,notes,source,{dark=false,backup=false}={}) {
  const s=pptx.addSlide();s._n=slides.length+1;s.background={color:dark?C.ink:C.white};
  slides.push({number:s._n,title,section,notes,source,backup});
  if(!dark) {
    text(s,section.toUpperCase(),0.6,0.30,12.1,0.26,12,{bold:true,color:C.transfer,charSpacing:1.1});
    text(s,title,0.6,0.83,12.1,0.9,32,{bold:true,color:C.ink});
  }
  s.addNotes(notes+'\n\nSumber pemeriksaan: '+source+
    '\nVersi ini menyajikan hasil secara kualitatif. Estimasi, interval dan denominator tersedia dalam presentasi lengkap. Tidak ada analisis penelitian baru.');
  return s;
}
function card(s,title,body,x,y,w,h,fill=C.pale,headSize=25,bodySize=22) {
  box(s,x,y,w,h,fill);
  text(s,title,x+0.23,y+0.18,w-0.46,0.7,headSize,{bold:true,color:C.ink});
  text(s,body,x+0.23,y+1.0,w-0.46,h-1.18,bodySize,{valign:'top'});
}
function note(s,body,y=6.48) {text(s,body,0.66,y,12.0,0.58,16,{color:C.muted});}
function table(s,heads,rows,widths,{x=.65,y=2.08,rowH=.72,headH=.72,font=21}={}) {
  let xx=x;
  heads.forEach((v,i)=>{box(s,xx,y,widths[i]-.035,headH,C.ink);text(s,v,xx+.13,y+.07,widths[i]-.3,headH-.14,font-1,{bold:true,color:C.white});xx+=widths[i];});
  rows.forEach((row,j)=>{let xx=x;row.forEach((v,i)=>{box(s,xx,y+headH+.04+j*rowH,widths[i]-.035,rowH-.035,j%2?C.light:C.pale);text(s,v,xx+.13,y+headH+.1+j*rowH,widths[i]-.3,rowH-.16,font);xx+=widths[i];});});
}
function footerSource(s,body){text(s,body,.68,7.12,11.9,.20,10,{color:C.muted});}

// The short hook is a question about intervention relevance, not map failure.
{
  const s=add('Beda Peta, Masih Relevan?','Pembuka',
    'Pertanyaannya adalah apakah ranking attention dari seed lain masih memilih bagian input yang berpengaruh pada model target. Relevansi di sini berarti respons terhadap intervensi yang ditentukan. Studi ini menemukan transfer parsial pada pasangan dengan kontrol lengkap. Perbandingan berlangsung dalam arsitektur dan dataset yang sama.',
    '../sources/primary_eight_strata.csv; manuscript V10 Introduction and Results',{dark:true});
  text(s,'Beda Peta,\nMasih Relevan?',.72,.80,8.1,1.92,50,{color:C.white,bold:true});
  text(s,'Audit transfer ranking attention\npada tulisan tangan Parkinsonian',.76,3.04,8.3,1.07,26,{color:'D2E6DD'});
  text(s,'Agriby Diandra Chaniago',.76,4.64,9,.45,24,{bold:true,color:C.white});
  text(s,'Purwono · Annastasya Nabila Elsa Wulandari · Alfian Ma’arif',.76,5.19,11.9,.44,16,{color:'D2E6DD'});
  text(s,'Universitas Harapan Bangsa · Universitas Ahmad Dahlan',.76,5.82,11.9,.40,16,{color:'D2E6DD'});
  text(s,'16 November 2026',.76,6.5,5.1,.39,19,{color:C.white});
  ['Peta berbeda','Target tetap','Relevansi diuji'].forEach((v,i)=>{box(s,9.6,1.12+i*1.4,3.0,1.0,'244B40');text(s,v,9.8,1.27+i*1.4,2.6,.67,24,{bold:true,color:C.white});if(i<2)arrow(s,11.1,2.17+i*1.4,0,.29,'9ED1BD');});
}
{
  const s=add('AUROC tinggi belum memvalidasi attention','Pertanyaan penelitian',
    'AUROC menilai pemisahan kelompok oleh prediksi. Kesamaan peta menilai reproduktibilitas attention. Relevansi ranking menilai respons saat bagian input terpilih diganti. Ketiganya adalah objek evaluasi berbeda. Studi ini tidak mengestimasi hubungan universal antara AUROC dan mutu penjelasan.',
    'Theissler et al.; Turbé et al.; ../sources/references.json');
  card(s,'Keluaran','Apakah model membedakan PD dan kontrol?',.65,2.0,3.85,2.56);
  card(s,'Peta','Apakah training ulang menyorot waktu yang sama?',4.75,2.0,3.85,2.56);
  card(s,'Ranking','Apakah bagian terpilih memengaruhi keluaran?',8.85,2.0,3.85,2.56);
  box(s,.65,5.06,12.05,.94,C.light);text(s,'Kalibrasi dan validitas fisiologis memerlukan pemeriksaan tersendiri.',.94,5.24,11.45,.56,25,{bold:true});
  note(s,'Skor prediksi yang tinggi menjawab pertanyaan yang berbeda dari kualitas penjelasan.');
}
{
  const s=add('Keluaran antar-seed tidak selalu serupa','Perubahan keluaran',
    'Ringkasan seluruh pasangan pada bank discovery menunjukkan variasi probabilitas dan logit. Karena itu, contoh keluaran serupa tidak dapat mewakili seluruh bank. Analisis kontinu dipertahankan sebagai analisis utama. Gate keluaran serupa hanya menjadi sensitivitas yang mencakup bagian terbatas dari pasangan.',
    '../sources/discovery_e1_ci.csv; ../sources/discovery_e1_sensitivity.csv');
  card(s,'Seluruh pasangan','Probabilitas dan logit berubah ketika seed training berganti.',.65,2.0,5.74,2.78);
  card(s,'Subset keluaran serupa','Sebagian kecil pasangan memenuhi kriteria yang lebih ketat.',6.77,2.0,5.9,2.78);
  arrow(s,6.43,3.4,.26);
  box(s,.65,5.15,12.02,.91,C.ink);text(s,'Contoh berikut menunjukkan keberadaan fenomena, bukan gambaran seluruh bank.',.93,5.31,11.45,.59,24,{bold:true,color:C.white});
  note(s,'Ringkasan dan batas subset tetap tersedia di versi presentasi dengan angka.');
}
{
  const ex=E.examples.find(r=>r.architecture==='gru');
  const s=add('Keluaran serupa dapat disertai peta berbeda','Contoh attention nyata',
    'Kurva ini berasal dari pasangan BiGRU pada rekaman discovery yang sama. Keluaran memenuhi gate ketat, tetapi distribusi attention berbeda. Contoh dipilih menurut urutan identitas di antara pasangan eligible, tanpa mengurutkan JSD atau keuntungan ranking. Identitas dan bobot asli disimpan dalam metadata. Grafik menampilkan bobot asli dengan label angka disembunyikan.',
    '../sources/attention_examples.json; ../sources/discovery_e1_sensitivity.csv');
  const data=[{name:'Seed target',labels:ex.alpha_a.map((_,i)=>String(i)),values:ex.alpha_a},
              {name:'Seed lain',labels:ex.alpha_b.map((_,i)=>String(i)),values:ex.alpha_b}];
  s.addChart(pptx.ChartType.line,data,{x:.65,y:2.0,w:8.6,h:3.85,
    chartColors:[C.own,C.source],showLegend:true,legendPos:'b',legendFontFace:'Arial',legendFontSize:17,
    showTitle:false,showValue:false,showBorder:false,showMarker:false,lineDataSymbol:'none',lineSize:1.8,
    catAxisHidden:true,valAxisHidden:true,catAxisLabelPos:'none',valAxisLabelPos:'none',
    catGridLine:{style:'none'},valGridLine:{style:'none'},valAxisMinVal:0,
    chartArea:{fill:{color:C.white},border:{color:C.white,pt:0}},plotArea:{fill:{color:C.white},border:{color:C.white,pt:0}}});
  charts.push({slide:s._n,chart:'ppt/charts/chart1.xml',source:'../sources/attention_examples.json',architecture:'gru',series:data,
    transform:'No scientific transformation. Original alpha values; numeric axes and labels hidden.'});
  text(s,'Awal rekaman',.95,5.83,2.6,.4,16,{color:C.muted});
  text(s,'Akhir rekaman',6.58,5.83,2.55,.4,16,{color:C.muted,align:'right'});
  text(s,'Bobot attention',.9,1.77,7,.29,16,{color:C.muted});
  card(s,'Keluaran serupa','Sorotan waktu\ntetap berbeda.',9.46,2.10,3.16,3.8,C.pale,25,25);
  note(s,'Contoh BiGRU dari bank discovery. Skala angka disembunyikan; kurva berasal dari artefak nyata.');
}
{
  const s=add('Apakah peta lain masih memilih bukti yang berguna?','Kesenjangan bukti',
    'Studi terdekat mengevaluasi dimensi penjelasan temporal, operator perturbasi, dan regularisasi penjelasan. Penelitian ini menguji transfer urutan pemilihan input pada target yang dibekukan. Kekhususannya terletak pada pertanyaan yang dapat dinilai dan kontrol alternatifnya. Pembandingan terbatas ini bukan klaim prioritas global.',
    '../notes/LITERATURE_COMPARISON.md; Bragança and Souto; Šimić et al.; Navakauskas and Dumpis');
  table(s,['Bukti yang tersedia','Tambahan evaluasi studi ini'],[
    ['Mutu penjelasan mempunyai beberapa dimensi','Peta dan relevansi ranking dinilai terpisah'],
    ['Respons bergantung pada cara input diganti','Besar dan bentuk intervensi dikontrol'],
    ['Penjelasan dapat berubah antar-training','Ranking diuji pada model target yang tetap']
  ],[5.5,6.55],{rowH:.94,font:22});
  note(s,'Nilai tambahnya adalah memeriksa kegunaan ranking yang berbeda pada rekaman dan target yang sama.');
}
{
  const s=add('Prosedur audit diuji pada akuisisi spiral berbeda','Data dan cakupan',
    'NewHandPD dan UCI-SST mempunyai instrumen dan representasi kanal berbeda. Model dilatih terpisah pada setiap dataset. UCI-SST adalah tugas spiral statis dan tidak digabung dengan tugas dinamis atau stabilitas. Transfer ranking tetap berlangsung dalam dataset dan arsitektur yang sama.',
    '../sources/newhandpd_record_manifest.csv; ../sources/uci395_record_manifest.csv; ../sources/PROVENANCE.json');
  card(s,'NewHandPD','Spiral dengan smart pen.\nKanal tekanan dan dinamika pena.',.65,2.03,5.8,2.76);
  card(s,'UCI-SST','Spiral statis dengan tablet.\nKoordinat dan kanal turunannya.',6.83,2.03,5.8,2.76);
  box(s,.65,5.17,11.98,.88,C.ink);text(s,'Training dilakukan kembali pada setiap dataset.',.94,5.35,11.38,.52,28,{color:C.white,bold:true});
  note(s,'Replikasi prosedur lintas akuisisi berbeda dari transfer satu model beku ke kohort baru.');
}
{
  const s=add('Bank baru memperluas bukti discovery','Riwayat dan desain',
    'Bank historis memberi konteks performa. Bank discovery mendasari audit awal dan contoh attention. Bank perluasan melaporkan evaluasi transfer pada seed baru. Hasil historis dan discovery tidak dipresentasikan seolah berasal dari bank perluasan. Tambahan seed meningkatkan cakupan variasi training, bukan jumlah subjek.',
    '../sources/evidence.json: design and cohort manifests');
  card(s,'Bank historis','Konteks AUROC dan analisis pendamping.',.65,2.12,3.85,2.62);
  card(s,'Bank discovery','Audit awal dan rancangan pengujian lanjutan.',4.75,2.12,3.85,2.62);
  card(s,'Bank perluasan','Evaluasi transfer dengan resep yang dikunci.',8.85,2.12,3.85,2.62);
  arrow(s,4.54,3.35,.18);arrow(s,8.64,3.35,.18);
  note(s,'Status eksploratori dan riwayat rencana tetap dilaporkan. Tidak ada klaim preregistrasi publik.');
}
{
  const s=add('Probabilitas dan attention berasal dari classifier','Alur model',
    'Sinyal diproses pada fold training, dibagi menjadi patch, lalu diteruskan ke encoder. Attention menimbang representasi temporal untuk menghasilkan prediksi. Encoder dapat menyebarkan informasi antarwaktu. Karena itu, bobot attention bukan anotasi lokasi gejala yang disupervisi.',
    'V10 Methods; ../sources/evidence.json: design');
  const xs=[.65,3.82,6.99,10.16];
  [['Sinyal','Preprocessing\nberdasarkan train fold'],['Patch temporal','BiGRU atau\nBiMamba-2'],['Attention','Bobot untuk\nrepresentasi waktu'],['Prediksi','Probabilitas\nkelompok PD']].forEach((v,i)=>{card(s,v[0],v[1],xs[i],2.2,2.52,2.65,C.pale,23,20);if(i<3)arrow(s,xs[i]+2.57,3.5,.50);});
  box(s,.65,5.25,12.03,.84,C.light);text(s,'Bobot besar mengarahkan pengujian. Respons intervensi menentukan relevansi operasional.',.93,5.37,11.47,.6,23,{bold:true});
  note(s,'Model ini dilatih untuk klasifikasi, bukan untuk melokalisasi gejala fisiologis.');
}
{
  const s=add('Target tetap ketika ranking sumber berubah','Eksperimen transfer ranking',
    'Pada rekaman yang sama, target diuji memakai ranking miliknya sendiri dan ranking dari seed lain. Arsitektur, dataset, fold, serta grid temporal harus cocok. Ranking menentukan patch input yang diganti. Bobot attention head tidak ditimpa. Forward target menghitung ulang responsnya sendiri.',
    '../sources/evidence.json: design; V10 fixed-target Methods');
  card(s,'Ranking target','Urutan patch dari attention milik model target.',.65,2.0,4.6,1.9,C.pale,25,20);
  card(s,'Ranking seed lain','Urutan patch dari model searsitektur pada rekaman yang sama.',.65,4.2,4.6,1.9,C.light,25,20);
  arrow(s,5.4,3.0,1.22,1.0);arrow(s,5.4,5.0,1.22,-1.0);
  card(s,'Model target dibekukan','Checkpoint dan normalizer yang sama.\nRespons dibandingkan secara berpasangan.',6.9,2.77,5.73,2.77,C.pale,28,23);
  note(s,'Yang dipindahkan adalah urutan pemilihan input, bukan bobot model atau attention internal.');
}
{
  const s=add('Intervensi kuat membutuhkan kontrol yang sepadan','Kontrol amplitudo dan geometri',
    'Setiap mask attention mempunyai kontrol acak yang dicocokkan sendiri. Jumlah patch, geometri run, dan besar perubahan input diukur dengan RMS. Penggantian menggunakan median train atau donor dari train fold. Pencarian kontrol tidak menggunakan respons model. Mask target dan mask transfer masih dapat berbeda dosis secara langsung.',
    'V10 Methods; ../sources/evidence.json: design and matching coverage');
  card(s,'Budget','Jumlah patch yang diganti dibuat sama.',.65,2.06,3.85,2.47);
  card(s,'Geometri','Panjang rangkaian patch dipertahankan.',4.75,2.06,3.85,2.47);
  card(s,'RMS','Besar perubahan input dicocokkan.',8.85,2.06,3.85,2.47);
  box(s,.65,5.02,12.05,1.02,C.light);text(s,'Median dan donor diuji terpisah. Kasus tanpa kontrol lengkap dilaporkan sebagai pengecualian.',.94,5.2,11.46,.66,24,{bold:true});
  note(s,'RMS merangkum besar perubahan input. Kontrol cocok untuk setiap mask, bukan otomatis antarmask.');
}
{
  const s=add('Keuntungan ranking dinilai terhadap kontrolnya','Cara membaca hasil',
    'Respons intervensi diringkas dari besar perubahan absolut logit di sepanjang budget. Keuntungan ranking adalah respons itu dikurangi respons kontrol RMS-matched. Efek transfer membandingkan keuntungan ranking target dengan keuntungan ranking sumber. Respons ini tidak mengukur validitas gejala dan tidak langsung membentuk leaderboard lintas arsitektur.',
    'V10 AOPC, G and D definitions');
  card(s,'Respons intervensi','Seberapa besar keluaran berubah setelah patch terpilih diganti?',.65,2.01,5.75,2.37);
  card(s,'Keuntungan ranking','Bagian respons yang melampaui kontrol sepadan.',6.83,2.01,5.8,2.37);
  box(s,.65,4.87,11.98,1.18,C.ink);text(s,'Bandingkan keuntungan ranking target dengan keuntungan ranking transfer.',.97,5.05,11.35,.8,28,{color:C.white,bold:true});
  note(s,'Logit adalah skor sebelum sigmoid. AOPC merangkum respons pada beberapa tingkat intervensi.');
}
function resultSlide(cohort,label) {
  const s=add('Relevansi ranking berpindah sebagian pada '+label,'Hasil transfer',
    'Pada rerata pasangan eligible, ranking dari seed lain melampaui kontrol RMS-matched. Ranking target mempunyai keuntungan yang lebih besar. Pola ini terdapat pada kedua arsitektur dan kedua replacement. Interval subjek bersyarat mendukung arah kontras tersebut. Kesimpulan merujuk pada support matching, bukan seluruh kemungkinan pasangan.',
    '../sources/primary_eight_strata.csv; ../sources/'+cohort+'_secondary_mean_contrasts.csv');
  table(s,['Model','Ranking transfer','Ranking target'],[
    ['BiGRU','Melampaui kontrol','Lebih unggul rata-rata'],
    ['BiMamba-2','Melampaui kontrol','Lebih unggul rata-rata']
  ],[2.6,4.65,4.8],{rowH:1.05,font:24,headH:.72});
  box(s,.65,5.4,12.05,.8,C.pale);text(s,'Pola rata-rata berlaku pada median dan donor dalam support yang lengkap.',.94,5.52,11.46,.56,24,{bold:true});
  note(s,'Keuntungan parsial berarti berguna pada intervensi ini, tanpa menyamakan transfer dengan kesetaraan.');
}
resultSlide('newhandpd','NewHandPD');
resultSlide('uci395','UCI-SST');
{
  const s=add('Matching menentukan cakupan kesimpulan','Cakupan hasil',
    'Sebagian pasangan tidak memperoleh kontrol lengkap dalam batas pencarian. Cakupan berbeda menurut dataset, arsitektur, replacement, dan perbandingan placebo. Kehadiran banyak subjek tidak berarti semua rekaman atau pasangan masuk. Karakteristik eligible dan excluded dapat berbeda. Batas ini berkaitan dengan populasi yang diwakili hasil.',
    '../sources/newhandpd_matching_coverage.csv; ../sources/uci395_matching_coverage.csv; eligible_excluded_characteristics');
  [['Pasangan direncanakan','Semua identitas yang masuk jadwal'],['Kontrol lengkap','Kasus yang berhasil dicocokkan'],['Analisis berpasangan','Irisan yang memenuhi syarat perbandingan']].forEach((v,i)=>{card(s,v[0],v[1],.65+i*4.1,2.13,3.85,2.5,C.pale,24,21);if(i<2)arrow(s,4.54+i*4.1,3.35,.17);});
  box(s,.65,5.1,12.05,.94,C.warnPale);text(s,'Kegagalan matching mempersempit cakupan. Itu bukan bukti bahwa matching mustahil.',.95,5.26,11.45,.65,25,{bold:true,color:C.warn});
  note(s,'Subset keluaran serupa mempunyai cakupan tersendiri dan tetap menjadi sensitivitas.');
}
{
  const s=add('Peta lebih berbeda berkaitan dengan efek transfer','Hubungan peta dan relevansi',
    'JSD mengukur perbedaan distribusi attention. Pada analisis target tetap, weighted Spearman menunjukkan asosiasi positif dengan besar selisih keuntungan transfer pada seluruh kondisi utama. Asosiasi ini merupakan hasil sekunder dengan interval nominal. Sebagian kaitannya dapat muncul dari konstruksi mask. Konsentrasi JSD tinggi membatasi penafsiran pada peta yang sangat mirip.',
    '../sources/newhandpd_jsd_associations.csv; ../sources/uci395_jsd_associations.csv');
  card(s,'Perbedaan peta','JSD merangkum perbedaan distribusi attention.',.65,2.18,5.2,2.36);
  card(s,'Besar efek transfer','Seberapa jauh keuntungan ranking target dan transfer berbeda.',7.4,2.18,5.2,2.36);
  arrow(s,6.0,3.35,1.2);
  box(s,.65,5.02,11.95,1.04,C.light);text(s,'Hubungan positif tidak otomatis berarti penjelasan memburuk.',.95,5.23,11.35,.61,29,{bold:true});
  note(s,'Analisis ini menjaga target tetap. Bridge discovery sebelumnya mengganti fungsi model sekaligus.');
}
{
  const s=add('Ranking nyata tidak selalu melampaui plasebo','Pembanding yang lebih menuntut',
    'Plasebo diperoleh melalui pergeseran atau permutasi bobot attention. Kandidat dipilih agar JSD mendekati pasangan nyata, tanpa melihat respons target. Ia mendapat kontrol sepadannya sendiri. Keunggulan ranking nyata muncul pada sebagian kondisi. Kondisi lain mempunyai interval yang mencakup nol. Cakupan matching juga berkurang.',
    '../sources/newhandpd_placebo_contrasts.csv; ../sources/uci395_placebo_contrasts.csv');
  card(s,'Peta dari seed lain','Ranking yang benar-benar dihasilkan oleh training lain.',.65,2.05,5.8,2.48);
  card(s,'Peta plasebo','Bobot digeser atau diacak pada perbedaan peta yang serupa.',6.85,2.05,5.8,2.48);
  box(s,.65,5.04,12.0,1.03,C.pale);text(s,'Perbandingan plasebo bergantung pada kondisi pengujian.',.94,5.23,11.42,.66,28,{bold:true});
  note(s,'Interval yang mencakup nol tidak membuktikan kesetaraan antara ranking nyata dan plasebo.');
}
{
  const s=add('Dosis langsung masih perlu diperhatikan','Sensitivitas intervensi',
    'Kontrol sepadan untuk setiap mask tidak otomatis menyamakan dosis ranking target dan transfer. Sensitivitas membatasi pasangan pada dosis yang lebih serupa. Subset ini menyempit. Pada donor UCI, interval kontras mentah mencakup nol pada kedua arsitektur. Temuan tersebut membatasi perluasan klaim dari hasil rata-rata utama.',
    '../sources/newhandpd_secondary_mean_contrasts.csv; ../sources/uci395_secondary_mean_contrasts.csv: dose scenarios');
  card(s,'Kontrol setiap mask','Ranking target dan transfer masing-masing mempunyai kontrol RMS.',.65,2.01,5.8,2.68);
  card(s,'Dosis antarranking','Dosis keduanya masih dapat berbeda secara langsung.',6.85,2.01,5.8,2.68,C.warnPale);
  box(s,.65,5.16,12.0,.94,C.pale);text(s,'Pembatasan dosis memperkecil support dan mengubah ketelitian beberapa kontras.',.95,5.32,11.4,.63,25,{bold:true});
  note(s,'Interpretasi mengikuti hasil per kondisi. Kontras utama tidak mengisolasi pengaruh ranking secara kausal.');
}
{
  const s=add('Rerata yang bertahan tetap menyisakan variasi','Training dan inferensi',
    'Leave-one-seed-out mempertahankan arah positif rerata efek transfer. Sebagian target BiGRU tetap mempunyai efek yang berlawanan. Ini menunjukkan variasi antar-training di balik rerata. Interval subjek bersyarat tidak mencakup seluruh variasi populasi seed atau pembagian fold baru. Pasangan berbagi model dan subjek.',
    '../sources/newhandpd_leave_one_seed_out.csv; ../sources/uci395_leave_one_seed_out.csv; target_seed_effects');
  card(s,'Ringkasan bank','Arah rerata bertahan ketika satu seed dikeluarkan.',.65,2.12,5.8,2.65);
  card(s,'Target individual','Sebagian target memberikan arah hasil yang berbeda.',6.85,2.12,5.8,2.65);
  box(s,.65,5.2,12.0,.88,C.ink);text(s,'Tambahan seed memperluas variasi training. Jumlah subjek tetap membatasi inferensi.',.94,5.36,11.42,.56,24,{color:C.white,bold:true});
  note(s,'Leave-one-seed-out merupakan diagnostik pengaruh seed, bukan interval populasi seed.');
}
{
  const s=add('Audit membantu menilai penjelasan secara bertahap','Manfaat praktis',
    'Pengguna model dapat memisahkan pemeriksaan prediksi, reproduktibilitas peta, dan respons intervensi. Kontrol amplitudo dan geometri membantu menguji alternatif penjelasan berupa perubahan input yang lebih besar. Transfer parsial memberi alasan untuk menguji ranking lain pada target yang sama saat debugging. Evaluasi klinis tetap memerlukan referensi tersendiri.',
    'V10 Discussion and Conclusion');
  card(s,'Evaluasi','Pisahkan mutu prediksi, peta, dan ranking.',.65,2.0,3.85,2.85);
  card(s,'Debugging','Uji ranking alternatif pada target yang sama.',4.75,2.0,3.85,2.85);
  card(s,'Perbandingan model','Periksa kontrol dan cakupan sebelum menilai penjelasan.',8.85,2.0,3.85,2.85);
  note(s,'Manfaatnya adalah prosedur audit yang dapat diperiksa, bukan cap universal bahwa attention baik atau buruk.');
}
{
  const s=add('Bukti mempunyai cakupan yang terdefinisi','Batas dan tindak lanjut',
    'Audit mencakup spiral, encoder yang dipilih, pembagian subjek tetap, dan replacement yang diuji. Beberapa analisis pendamping berasal dari bank historis atau discovery. Rencana tidak didaftarkan publik. Analisis tambahan dan sensitivitas tetap eksploratori. Replikasi split, kohort lebih besar, atribusi lain, dan rujukan fisiologis menjawab pertanyaan berikutnya.',
    'V10 Discussion; ../notes/CLAIM_AUDIT.md');
  table(s,['Sudah diperiksa','Pertanyaan berikutnya'],[
    ['Transfer ranking pada target tetap','Kohort dan pembagian subjek lain'],
    ['Kontrol RMS serta geometri temporal','Replacement dan attribution lain'],
    ['Pengaruh seed dalam bank yang tersedia','Ketelitian lintas-training yang lebih luas'],
    ['Respons model terhadap intervensi','Validitas terhadap gejala fisiologis']
  ],[6.0,6.05],{rowH:.73,font:21});
  note(s,'Temuan mengikuti dataset, training, intervensi, dan pasangan eligible yang benar-benar diuji.');
}
{
  const s=add('Relevansi berpindah, tetapi hanya sebagian','Kesimpulan',
    'Ranking dari seed lain mempertahankan keuntungan rata-rata terhadap kontrol pada pasangan eligible. Ranking target lebih unggul rata-rata. Besar perbedaan peta berkaitan dengan besar efek transfer, sedangkan pembanding plasebo memberi hasil yang bergantung kondisi. Kesimpulan ini menyangkut intervensi yang diuji, bukan validitas fisiologis atau mutu penjelasan universal.',
    '../sources/primary_eight_strata.csv; secondary_mean_contrasts; jsd_associations; placebo_contrasts',{dark:true});
  text(s,'Peta berbeda.\nRelevansi berpindah sebagian.',.75,.94,11.8,1.61,43,{color:C.white,bold:true});
  [['Ranking transfer','Tetap melampaui kontrol rata-rata.'],['Ranking target','Memiliki keuntungan rata-rata yang lebih besar.'],['Perbedaan peta','Berkaitan dengan besar efek, dengan batas kondisi.']].forEach((v,i)=>{box(s,.77,3.0+i*1.07,11.8,.89,'244B40');text(s,v[0],1.0,3.15+i*1.07,3.04,.57,23,{color:C.white,bold:true});text(s,v[1],4.16,3.15+i*1.07,8.1,.57,23,{color:'D2E6DD'});});
  text(s,'AUROC, kemiripan peta, dan relevansi ranking menjawab pertanyaan yang berbeda.',.8,6.55,11.74,.52,23,{color:C.white,bold:true});
}
{
  const s=add('Magnitude juga dapat melampaui attention','Cadangan · pembanding ranking',
    'Pembanding magnitude menggunakan besar perubahan input untuk mengurutkan patch. Pada discovery, BiGRU lebih menguntungkan attention pada donor dan median. BiMamba lebih menguntungkan magnitude pada donor. Pada median BiMamba, dukungan interval bergantung support penuh atau irisan E2 lengkap. Magnitude belum mendapat dose dan geometri yang disamakan. Ia bukan bukti arsitektur paling faithful.',
    '../sources/discovery_amplitude_ci.csv; V10 magnitude comparator tables',{backup:true});
  table(s,['Kondisi discovery','Temuan kualitatif'],[
    ['BiGRU · donor dan median','Attention unggul terhadap magnitude'],
    ['BiMamba-2 · donor','Magnitude unggul terhadap attention'],
    ['BiMamba-2 · median','Dukungan interval berubah menurut support']
  ],[5.2,6.85],{rowH:1.0,font:22});
  note(s,'Pembanding magnitude belum menyamakan dosis dan geometri. Kedua support dilaporkan pada versi lengkap.');
}
{
  const s=add('Hasil pendamping memberi konteks yang berbeda','Cadangan · performa dan probabilitas',
    'Bank historis menunjukkan penurunan AUROC BiMamba yang lebih kecil secara deskriptif setelah retraining terpisah. Interval kontras masih mencakup nol. Bank discovery mendukung Brier lebih rendah untuk BiMamba, sedangkan ECE utama masih tidak pasti. Brier mengukur kualitas probabilitas secara keseluruhan. Kontrol usia dan baseline klasik historis mempunyai batas protokolnya sendiri.',
    '../sources/historical_auroc.csv; historical_auroc_contrast.csv; calibration_contrasts.csv; ../notes/CLAIM_AUDIT.md',{backup:true});
  card(s,'AUROC historis','Konsistensi BiMamba-2 bersifat deskriptif setelah retraining.',.65,2.02,3.85,2.81);
  card(s,'Brier discovery','Mendukung kualitas probabilitas yang lebih baik untuk BiMamba-2.',4.75,2.02,3.85,2.81);
  card(s,'Interpretasi','Performa, kalibrasi, dan fisiologi dinilai terpisah.',8.85,2.02,3.85,2.81);
  note(s,'Hasil ini bukan evaluasi kalibrasi bank perluasan dan bukan keberhasilan transfer satu model antar-kohort.');
}
{
  const s=add('Istilah yang perlu dibedakan','Cadangan · panduan interpretasi',
    'Reproduktibilitas peta adalah kesamaan attention antar-training. Relevansi intervensi adalah respons model terhadap penggantian input yang ditentukan. Validitas fisiologis membutuhkan rujukan gejala yang tepat. Kalibrasi adalah hubungan probabilitas dengan frekuensi kejadian. Selang yang mencakup nol bukan bukti ekuivalensi atau hubungan nol.',
    'V10 Methods and Discussion; ../sources/references.json',{backup:true});
  table(s,['Istilah','Makna pada penelitian ini'],[
    ['Seed','Kondisi acak untuk mengulang training'],
    ['Ranking attention','Urutan patch dari bobot attention'],
    ['Relevansi intervensi','Respons yang melampaui kontrol sepadan'],
    ['JSD','Perbedaan distribusi bobot attention'],
    ['Kalibrasi','Kecocokan probabilitas dengan frekuensi kejadian']
  ],[4.1,7.95],{rowH:.64,font:21});
  note(s,'Peta yang berbeda belum tentu peta yang buruk. Interpretasi mengikuti pengujian yang benar-benar dilakukan.');
}
{
  const s=add('Penelitian terdekat memberi pembanding protokol','Cadangan · literatur inti',
    'Daftar ini merupakan pembanding metodologis, bukan leaderboard performa handwriting. Judul, jurnal, tahun dan DOI lengkap terdapat pada sumber presentasi dengan angka. Tidak ada klaim bahwa ketiganya sepenuhnya menyamakan protokol penelitian ini.',
    '../sources/references.json; ../notes/LITERATURE_COMPARISON.md',{backup:true});
  card(s,'Bragança dan Souto','UTS-XAI\nEvaluasi beberapa dimensi penjelasan temporal.\nScientific Reports',.65,2.0,3.85,3.35,C.pale,24,21);
  card(s,'Šimić dan kolega','Operator perturbasi dan ukuran region.\nScientific Reports',4.75,2.0,3.85,3.35,C.pale,24,21);
  card(s,'Navakauskas dan Dumpis','Regularisasi penjelasan temporal.\nElectronics',8.85,2.0,3.85,3.35,C.pale,24,21);
  note(s,'Kekhususan studi terletak pada pengujian transfer berpasangan, target tetap, dan kontrol intervensinya.');
}
{
  const s=add('Rujukan dan angka lengkap tetap dapat diperiksa','Cadangan · sumber dan penggunaan',
    'Presentasi ini adalah pendamping kualitatif. Seluruh angka, confidence interval, denominator, definisi gate, dan rujukan lengkap tetap tersedia dalam versi kuantitatif pada folder induk. Kurva contoh menggunakan bobot attention asli. Diagram lainnya bersifat konseptual dan tidak menyimulasikan data penelitian. Tidak ada eksperimen atau estimator yang dijalankan.',
    '../README.md; ../CATATAN_PEMBICARA.md; ../sources/PROVENANCE.json; Q1-amin/manuscript_array_latex_v10/README.md',{backup:true});
  card(s,'Presentasi lengkap','Gunakan untuk memeriksa estimasi, interval, dan cakupan.',.65,2.0,5.8,2.67);
  card(s,'Manuskrip terbaru','Judul diselaraskan. Angka dan batas ilmiah dipertahankan.',6.85,2.0,5.8,2.67);
  box(s,.65,5.15,12.0,.88,C.light);text(s,'Versi tanpa angka hasil memudahkan penyampaian cerita; ia tidak menggantikan bukti kuantitatif.',.94,5.31,11.43,.57,24,{bold:true});
  note(s,'Nama resmi model dan tanggal acara tetap ditampilkan. Tidak ada statistik hasil pada slide.');
}

fs.mkdirSync(path.join(ROOT,'review'),{recursive:true});
fs.writeFileSync(path.join(ROOT,'SLIDE_PLAN.json'),JSON.stringify({title:pptx.title,main_slides:21,total_slides:slides.length,slides},null,2)+'\n');
fs.writeFileSync(path.join(ROOT,'review/SHAPE_LAYOUT.json'),JSON.stringify(shapes,null,2)+'\n');
fs.writeFileSync(path.join(ROOT,'review/CHART_BINDINGS.json'),JSON.stringify(charts,null,2)+'\n');
fs.writeFileSync(path.join(ROOT,'CATATAN_PEMBICARA.md'),'# Beda Peta, Masih Relevan?\n\nVersi tanpa angka hasil. Tanggal acara dan nama resmi model dipertahankan.\n\n'+slides.map(s=>'## '+s.number+'. '+s.title+'\n\n'+s.notes+'\n\nSumber: `'+s.source+'`\n').join('\n'));
pptx.writeFile({fileName:path.join(ROOT,'presentasi_attention_16nov2026_tanpa_angka.pptx')});
