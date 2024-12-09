.equ switch, 0xff200050
.equ led, 0xff200000

.global _start
_start:
	
	
	ldr r9, =switch
	ldr r2, =led
	
	//mov r2, #9
	//mov r3, #5
	
	ldr r1, [r9]
	str r1, [r2] // use push button to take input
	
	ldr r1, [r9]
	str r1, [r3]
	
	
	
	bl fun_sub
	mov r7, r6
	b next_num
	
fun_sub:
	push {lr}
	bl check_equ
	cmp r0, #0
	beq end
	sub r6, r2, r3
	pop {lr}
	mov pc, lr
	
	
	
	
	check_equ:
		cmp r2, r3
		beq EQU
		mov r0, #1
		mov pc, lr
		EQU:
		mov r0, #1
		mov pc, lr
		
	
	next_num:
	
	//mov r2, #9
	//mov r3, #5
	
	ldr r1, [r9]
	str r1, [r2] // use push button to take input
	
	ldr r1, [r9]
	str r1, [r3]
	
	bl fun_sub
	mul r8, r6, r7
	
	
	end:
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	