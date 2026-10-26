export const WHISPER_LANGUAGES = [
  {code:'auto',name:'Auto Detect'},{code:'af',name:'Afrikaans'},{code:'ar',name:'Arabic'},
  {code:'hy',name:'Armenian'},{code:'az',name:'Azerbaijani'},{code:'be',name:'Belarusian'},
  {code:'bs',name:'Bosnian'},{code:'bg',name:'Bulgarian'},{code:'ca',name:'Catalan'},
  {code:'zh',name:'Chinese'},{code:'hr',name:'Croatian'},{code:'cs',name:'Czech'},
  {code:'da',name:'Danish'},{code:'nl',name:'Dutch'},{code:'en',name:'English'},
  {code:'et',name:'Estonian'},{code:'fi',name:'Finnish'},{code:'fr',name:'French'},
  {code:'gl',name:'Galician'},{code:'de',name:'German'},{code:'el',name:'Greek'},
  {code:'he',name:'Hebrew'},{code:'hi',name:'Hindi'},{code:'hu',name:'Hungarian'},
  {code:'is',name:'Icelandic'},{code:'id',name:'Indonesian'},{code:'it',name:'Italian'},
  {code:'ja',name:'Japanese'},{code:'kn',name:'Kannada'},{code:'kk',name:'Kazakh'},
  {code:'ko',name:'Korean'},{code:'lv',name:'Latvian'},{code:'lt',name:'Lithuanian'},
  {code:'mk',name:'Macedonian'},{code:'ms',name:'Malay'},{code:'ml',name:'Malayalam'},
  {code:'mr',name:'Marathi'},{code:'mi',name:'Maori'},{code:'ne',name:'Nepali'},
  {code:'no',name:'Norwegian'},{code:'fa',name:'Persian'},{code:'pl',name:'Polish'},
  {code:'pt',name:'Portuguese'},{code:'pa',name:'Punjabi'},{code:'ro',name:'Romanian'},
  {code:'ru',name:'Russian'},{code:'sr',name:'Serbian'},{code:'sk',name:'Slovak'},
  {code:'sl',name:'Slovenian'},{code:'es',name:'Spanish'},{code:'sw',name:'Swahili'},
  {code:'sv',name:'Swedish'},{code:'tl',name:'Tagalog'},{code:'ta',name:'Tamil'},
  {code:'te',name:'Telugu'},{code:'th',name:'Thai'},{code:'tr',name:'Turkish'},
  {code:'uk',name:'Ukrainian'},{code:'ur',name:'Urdu'},{code:'vi',name:'Vietnamese'},
  {code:'cy',name:'Welsh'},{code:'yi',name:'Yiddish'},{code:'yo',name:'Yoruba'},
  {code:'jw',name:'Javanese'},{code:'lo',name:'Lao'},{code:'la',name:'Latin'},
  {code:'ht',name:'Haitian Creole'},{code:'ba',name:'Bashkir'},{code:'eu',name:'Basque'},
  {code:'bn',name:'Bengali'},{code:'my',name:'Burmese'},{code:'km',name:'Khmer'},
  {code:'tt',name:'Tatar'},{code:'haw',name:'Hawaiian'},{code:'fo',name:'Faroese'},
  {code:'mg',name:'Malagasy'},{code:'mt',name:'Maltese'},{code:'lb',name:'Luxembourgish'},
  {code:'br',name:'Breton'},{code:'sa',name:'Sanskrit'},{code:'si',name:'Sinhala'},
  {code:'nn',name:'Nynorsk'},{code:'sn',name:'Shona'},{code:'sd',name:'Sindhi'},
  {code:'gu',name:'Gujarati'},{code:'as',name:'Assamese'},{code:'su',name:'Sundanese'},
]
export const TARGET_LANGUAGES = [
  {code:'en',name:'English'},{code:'te',name:'Telugu'},{code:'hi',name:'Hindi'},
  {code:'ta',name:'Tamil'},{code:'kn',name:'Kannada'},{code:'ml',name:'Malayalam'},
  {code:'gu',name:'Gujarati'},{code:'mr',name:'Marathi'},{code:'pa',name:'Punjabi'},
  {code:'bn',name:'Bengali'},{code:'ur',name:'Urdu'},{code:'zh',name:'Chinese (Simplified)'},
  {code:'ja',name:'Japanese'},{code:'ko',name:'Korean'},{code:'ar',name:'Arabic'},
  {code:'fr',name:'French'},{code:'de',name:'German'},{code:'es',name:'Spanish'},
  {code:'pt',name:'Portuguese'},{code:'ru',name:'Russian'},{code:'it',name:'Italian'},
  {code:'nl',name:'Dutch'},{code:'pl',name:'Polish'},{code:'tr',name:'Turkish'},
  {code:'vi',name:'Vietnamese'},{code:'th',name:'Thai'},{code:'id',name:'Indonesian'},
  {code:'ms',name:'Malay'},{code:'sv',name:'Swedish'},{code:'he',name:'Hebrew'},
  {code:'uk',name:'Ukrainian'},{code:'ro',name:'Romanian'},{code:'el',name:'Greek'},
  {code:'fa',name:'Persian'},{code:'sw',name:'Swahili'},{code:'af',name:'Afrikaans'},
]

// Google Input Tools transliteration "itc" codes — type Latin/English letters
// phonetically, get native-script suggestions. Only languages with a working
// itc code are listed; languages not here just don't get the transliteration
// toggle (plain typing still works everywhere).
// Google's phonetic transliteration only covers alphabet-based scripts.
// Japanese, Chinese and Korean are NOT here on purpose: they use a
// dictionary/candidate-based IME, a completely different mechanism — this
// simple phonetic scheme doesn't work for them at all.
export const TRANSLITERATION_ITC = {
  hi: 'hi-t-i0-und', te: 'te-t-i0-und', ta: 'ta-t-i0-und', kn: 'kn-t-i0-und',
  ml: 'ml-t-i0-und', gu: 'gu-t-i0-und', mr: 'mr-t-i0-und', pa: 'pa-t-i0-und',
  bn: 'bn-t-i0-und', ur: 'ur-t-i0-und', ne: 'ne-t-i0-und', si: 'si-t-i0-und',
  sa: 'sa-t-i0-und', as: 'as-t-i0-und',
  ar: 'ar-t-i0-und', fa: 'fa-t-i0-und', ru: 'ru-t-i0-und', el: 'el-t-i0-und',
  sr: 'sr-t-i0-und', am: 'am-t-i0-und', ti: 'ti-t-i0-und', he: 'he-t-i0-und',
}

// Languages that need a real dictionary-based IME rather than the scheme
// above — used only to show an honest note instead of silently doing nothing.
export const IME_ONLY_LANGUAGES = { ja: 'Japanese', zh: 'Chinese', ko: 'Korean' }

// BCP-47 locale tags for the browser SpeechRecognition (mic dictation) API.
// Falls back to the plain language code if not listed, which works for most
// major languages already.
export const SPEECH_RECOGNITION_LOCALE = {
  en:'en-US', te:'te-IN', hi:'hi-IN', ta:'ta-IN', kn:'kn-IN', ml:'ml-IN',
  gu:'gu-IN', mr:'mr-IN', pa:'pa-IN', bn:'bn-IN', ur:'ur-IN', zh:'zh-CN',
  ja:'ja-JP', ko:'ko-KR', ar:'ar-SA', fr:'fr-FR', de:'de-DE', es:'es-ES',
  pt:'pt-PT', ru:'ru-RU', it:'it-IT', nl:'nl-NL', pl:'pl-PL', tr:'tr-TR',
  vi:'vi-VN', th:'th-TH', id:'id-ID', ms:'ms-MY', sv:'sv-SE', he:'he-IL',
  uk:'uk-UA', ro:'ro-RO', el:'el-GR', fa:'fa-IR', sw:'sw-KE', af:'af-ZA',
}