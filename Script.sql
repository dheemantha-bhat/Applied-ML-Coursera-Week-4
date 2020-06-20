### sql script ####
select docdt,count(distinct update_lid) from


(select update_lid,docdt,dense_rank() over (partition by update_lid order by docdt) rk from transaction_w
where  store_tag in ('WOT','LWOT')) c

where 
 UPDATE_LID !='NULL'  AND UPDATE_LID IS NOT NULL
AND UPDATE_LID <> ' '
AND update_lid not in ('0','00','000','0000','00000','000000','000000','0000000','00000000','000000000')
AND DOCDT > '2018-11-01' and rk=1
GROUP BY 1

select count(distinct update_lid) from transaction_w
where docdt between '2019-06-01' and '2019-06-30' and  store_tag in ('WOT','LWOT')