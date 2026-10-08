const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function setup(fetch) {
  const elements = {};
  const document = { addEventListener() {}, getElementById(id) {
    if (!elements[id]) elements[id] = { hidden: false, disabled: false, textContent: '',
      listeners: {}, attributes: {}, classList: { add() {}, remove() {} },
      addEventListener(type, handler) { this.listeners[type] = handler; },
      scrollIntoView() { this.scrolledIntoView = true; },
      pause() {}, removeAttribute(name) { delete this.attributes[name]; },
      setAttribute(name, value) { this.attributes[name] = value; } };
    return elements[id];
  }};
  const context = vm.createContext({ document, fetch, addEventListener() {}, URL: {createObjectURL: ()=>'blob:preview',revokeObjectURL(){}},
    FormData: class { constructor() {this.items=[]} append(...args) {this.items.push(args)} } });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/recorder.js'),'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8'), context);
  const select = (file) => {context.file = file; vm.runInContext('selectFile(file)',context)};
  const submit = () => elements['upload-form'].listeners.submit({preventDefault(){}});
  return {elements,select,submit};
}
const file = {name:'test.wav',size:1000};
const success = {prediction:{display_label:'Singapore accent'},audio:{duration_seconds:3,classified_duration_seconds:3},
  model_version:'e1-svm39-mvp-1',warnings:[]};

test('selection clears stale result and validates file', () => {
  const {elements,select} = setup();
  select(file);
  assert.equal(elements.submit.disabled,false);
  assert.equal(elements.filename.textContent,'test.wav');
  assert.equal(elements.result.hidden,true);
  select({name:'bad.mp3',size:100});
  assert.equal(elements.submit.disabled,true);
  assert.equal(elements.error.hidden,false);
  select(file);
  assert.equal(elements.error.hidden,true);
});

test('loading prevents selection changes; success renders label and restores controls', async () => {
  let resolve;
  const {elements,select,submit} = setup(() => new Promise(r=>resolve=r));
  select(file);
  const request=submit();
  assert.equal(elements.submit.disabled,true);
  assert.equal(elements['audio-file'].disabled,true);
  assert.equal(elements.loading.hidden,false);
  select({name:'other.wav',size:100});
  assert.equal(elements.filename.textContent,'test.wav');
  resolve({ok:true,json:async()=>success});
  await request;
  assert.equal(elements.prediction.textContent,'Singapore accent');
  assert.equal(elements.result.hidden,false);
  assert.equal(elements['result-card'].scrolledIntoView,true);
  assert.equal(elements.loading.hidden,true);
  assert.equal(elements.submit.disabled,false);
  assert.equal(elements['result-card'].attributes['aria-busy'],'false');
  assert.equal(elements.duration.textContent,'3.00 s');
  select({name:'other.wav',size:100});
  assert.equal(elements.duration.textContent,'—');
  assert.equal(elements.result.hidden,true);
});

test('API failure hides result and permits retry', async () => {
  let count=0;
  const {elements,select,submit}=setup(async()=> ++count === 1 ?
    {ok:false,json:async()=>({error:{message:'The recording is silent.'}})} : {ok:true,json:async()=>success});
  select(file); await submit();
  assert.equal(elements.error.textContent,'The recording is silent.');
  assert.equal(elements.result.hidden,true);
  assert.equal(elements.submit.disabled,false);
  await submit();
  assert.equal(elements.result.hidden,false);
  assert.equal(elements.error.hidden,true);
});

test('network failure produces readable error', async () => {
  const {elements,select,submit}=setup(async()=>{throw new Error('Failed to fetch')});
  select(file); await submit();
  assert.match(elements.error.textContent,/Cannot reach the server/);
  assert.equal(elements.loading.hidden,true);
  assert.equal(elements.submit.disabled,false);
});
