import { assetUrl, element, log, selectField, sourceLink, startShowcase } from './ui.js';

const names={bacterial_pneumonia:'Bacterial pneumonia',normal:'Normal',viral_pneumonia:'Viral pneumonia'};

function makeMatrix(model, state, choose) {
  const table=element('table','sc-matrix');
  table.append(element('caption','','Rows are dataset classes; columns are model predictions.'));
  const head=element('thead');
  const heading=element('tr');
  heading.append(element('th','','True / predicted'));
  for(const name of model.classes)heading.append(element('th','',names[name]));
  head.append(heading);table.append(head);
  const body=element('tbody');
  model.matrix.forEach((row,rowIndex)=>{
    const total=row.reduce((sum,value)=>sum+value,0);
    const tr=element('tr');
    const rowHeading=element('th','',names[model.classes[rowIndex]]);
    rowHeading.scope='row';tr.append(rowHeading);
    row.forEach((count,columnIndex)=>{
      const td=element('td');
      const button=element('button','',state.rates?`${(count/total*100).toFixed(1)}%`:String(count));
      button.type='button';
      button.style.setProperty('--cell-strength',`${12+count/total*30}%`);
      button.setAttribute('aria-pressed',String(state.row===rowIndex&&state.column===columnIndex));
      button.setAttribute('aria-label',`${names[model.classes[rowIndex]]} predicted as ${names[model.classes[columnIndex]]}: ${count} of ${total}`);
      button.append(element('small','',state.rates?`${count} of ${total}`:`${(count/total*100).toFixed(1)}% of row`));
      button.addEventListener('click',()=>choose(rowIndex,columnIndex));
      td.append(button);tr.append(td);
    });
    body.append(tr);
  });
  table.append(body);
  return table;
}

function makeDetails(model, state, details) {
  const count=model.matrix[state.row][state.column];
  const total=model.matrix[state.row].reduce((sum,value)=>sum+value,0);
  const trueClass=model.classes[state.row];
  const predictedClass=model.classes[state.column];
  const text=state.row===state.column?`${count} of ${total} ${names[trueClass].toLowerCase()} examples were assigned their dataset class.`:`${count} of ${total} ${names[trueClass].toLowerCase()} examples were classified as ${names[predictedClass].toLowerCase()}.`;
  details.replaceChildren(element('h3','','Read the selected cell'),element('p','',text));
  const report=model.report[trueClass];
  const list=element('dl');
  for(const [label,key]of [['Precision','precision'],['Recall','recall'],['F1','f1-score']])list.append(element('dt','',label),element('dd','',`${(report[key]*100).toFixed(1)}%`));
  details.append(list,sourceLink(model.source,'Stored evaluation report'));
}

function renderMatrix(root,data,base) {
  const controls=root.querySelector('.sc-controls');
  const stage=root.querySelector('.sc-stage');
  const details=root.querySelector('.sc-details');
  const live=root.querySelector('[role="status"]');
  const state={model:0,rates:false,row:2,column:0};
  const modelField=selectField('Model',data.models.map((model,index)=>[index,model.title]),'model');
  const displayField=selectField('Display',[[0,'Counts'],[1,'Share of true-class row']],'display');
  const classField=selectField('True class',data.models[0].classes.map((name,index)=>[index,names[name]]),'class');
  classField.select.value='2';
  controls.replaceChildren(modelField.wrapper,displayField.wrapper,classField.wrapper);
  const update=()=>{
    const model=data.models[state.model];
    const total=model.matrix.flat().reduce((sum,value)=>sum+value,0);
    const correct=model.matrix.reduce((sum,row,index)=>sum+row[index],0);
    const wrap=element('div','sc-matrix-wrap');
    wrap.append(makeMatrix(model,state,(row,column)=>{state.row=row;state.column=column;classField.select.value=String(row);update();}));
    stage.replaceChildren(wrap);
    makeDetails(model,state,details);
    live.textContent=`${model.title}: ${correct} correct and ${total-correct} errors in the stored ${total}-image test report.`;
    log('matrix:update',{model:model.id,row:state.row,column:state.column,rates:state.rates});
  };
  modelField.select.addEventListener('change',()=>{state.model=Number(modelField.select.value);update();});
  displayField.select.addEventListener('change',()=>{state.rates=displayField.select.value==='1';update();});
  classField.select.addEventListener('change',()=>{state.row=Number(classField.select.value);state.column=state.row;update();});
  update();
  const gallery=root.querySelector('.sc-sample-gallery');
  if(gallery)for(const sample of data.samples){const figure=element('figure');const image=element('img');image.src=assetUrl(sample.image,base);image.alt=`Source sample labelled ${names[sample.label]}`;figure.append(image,element('figcaption','',`Dataset label: ${names[sample.label]}`));gallery.append(figure);}
}

startShowcase('matrix',renderMatrix,new URL('./data.json',import.meta.url));
